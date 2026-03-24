import pandas as pd
import google.generativeai as genai
import time
import os
import re
import json
import asyncio
from tqdm import tqdm

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================
try:
    from kaggle_secrets import UserSecretsClient
    api_key = UserSecretsClient().get_secret("GOOGLE_API_KEY")
except ImportError:
    api_key = os.environ.get("GOOGLE_API_KEY", "[TU_API_KEY_AQUI]")

genai.configure(api_key=api_key)

model_name = 'gemini-3-flash-preview'  # Ajustar según tu modelo disponible
model = genai.GenerativeModel(model_name)

TRAIN_FILE = "/kaggle/input/competitions/deep-past-initiative-machine-translation/train.csv"
LEXICON_FILE = "/kaggle/input/competitions/deep-past-initiative-machine-translation/OA_Lexicon_eBL.csv"
OUTPUT_FILE = "./gemini_back_translated_train.csv"
VARIATIONS = 3  # Cuántas variaciones acadias generar por oración

# ============================================================
# PARÁMETROS DE PARALELISMO
# ============================================================
BATCH_SIZE = 10        # Docs por llamada API (menos porque genera 3 variaciones)
MAX_CONCURRENT = 5     # Llamadas API simultáneas
SAVE_EVERY = 50        # Guardar checkpoint cada N traducciones

# ============================================================
# 2. DICCIONARIO RAG
# ============================================================
def load_lexicon(path):
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        if 'form' in df.columns and 'norm' in df.columns:
            df = df.dropna(subset=['form', 'norm'])
            return {str(k).strip(): str(v).strip()
                    for k, v in zip(df['form'], df['norm'])}
    except Exception as e:
        print(f"Error diccionario: {e}")
    return {}

def get_hints_for_chunk(text, lexicon):
    if not lexicon:
        return ""
    words = text.replace('[', '').replace(']', '').split()
    hints = {w: lexicon[w] for w in words if w in lexicon and lexicon[w] != w}
    if not hints:
        return ""
    return "\nDICTIONARY HINTS:\n" + "".join(f"- {k} -> {v}\n" for k, v in hints.items())


# ============================================================
# 3. FEW-SHOT EXAMPLES (Invertidos: Inglés → Acadio)
# ============================================================
BACK_TRANSLATION_EXAMPLES = """
--- Example 1 ---
English: Itūr-ilī has received one textile of ordinary quality.
Transliteration: 1 TÚG ša qá-tim i-tur₄-DINGIR il₅-qé

--- Example 2 ---
English: Seal of Mannum-balum-Aššur son of Ṣilli-Adad. Puzur-Aššur son of Ataya owes 22 shekels of good silver to Ali-ahum.
Transliteration: KIŠIB ma-nu-ba-lúm-a-šur DUMU ṣí-lá-{d}IM 0.3333 ma-na 2 GÍN KÙ.BABBAR SIG₅ i-ṣé-er PÚZUR₄-a-šur DUMU a-ta-a a-lá-hu-um i-šu

--- Example 3 ---
English: From Šukkutum to Ištar-lamassī: If you are truly my sister, then encourage her. Do not fear.
Transliteration: um-ma šu-ku-tum-ma a-na IŠTAR-lá-ma-sí qí-bi₄-ma šu-ma a-ha-tí a-ta li-ba-am dì-ni-ší-im lá ta-ha-da-ar

--- Example 4 ---
English: 17 shekels of silver, the price of a donkey, 30 minas of copper, 2 jars of cumin
Transliteration: 17 GÍN KÙ.BABBAR ší-im e-ma-ri-im 30 ma-na URUDU 2 kà-ar-pát kà-mu-nu
"""


# ============================================================
# 6. PROMPT + API CALLS
# ============================================================
def build_batch_prompt(batch_items: list) -> str:
    docs = []
    for item in batch_items:
        doc = {"id": item['oare_id'], "english": item['english']}
        docs.append(doc)
    
    return f"""You are a world-class Assyriologist specializing in Old Assyrian (OA) Akkadian from Kanesh (Kültepe), ca. 1950-1850 BC.

TASK: Convert {len(batch_items)} English translations BACK into Old Assyrian Akkadian transliteration.
For EACH input, generate {VARIATIONS} different plausible transliteration variants.

CONVENTIONS (English → OA Transliteration):
1. "Seal of" → KIŠIB. "son of" → DUMU.
2. "silver"→KÙ.BABBAR, "gold"→KÙ.GI, "copper"→URUDU, "tin"→AN.NA, "textile(s)"→TÚG, "shekel(s)"→GÍN, "mina(s)"→ma-na, "talent(s)"→GÚ
3. Personal names → syllabic OA spelling: "Puzur-Aššur" → pu-zur₄-a-šur
4. Use OA syllabic conventions with hyphens between signs.
5. Letters should start with: um-ma [sender]-ma a-na [recipient] qí-bi₄-ma
6. Generate PLAUSIBLE Old Assyrian (ca. 1950 BC) transliteration, NOT Standard Babylonian.
7. Each variant should use slightly different sign choices or word order where grammatically valid.

EXAMPLES:
{BACK_TRANSLATION_EXAMPLES}

INPUT:
{json.dumps(docs, indent=2, ensure_ascii=False)}

Return ONLY a JSON array. Each object: {{"id": "...", "variants": ["variant1", "variant2", "variant3"]}}. No markdown.

JSON:"""

class IncompleteBatchError(Exception):
    pass

async def translate_batch(batch_items: list, semaphore: asyncio.Semaphore, retries: int = 3) -> dict:
    """Back-traduce un batch con semáforo para controlar concurrencia."""
    async with semaphore:
        prompt = build_batch_prompt(batch_items)
        
        for attempt in range(retries):
            try:
                response = await model.generate_content_async(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        response_mime_type="application/json",
                    )
                )
                await asyncio.sleep(0.5)
                
                parsed = json.loads(response.text.strip())
                if isinstance(parsed, dict):
                    parsed = [parsed]
                
                # Parse: each item should have "id" and "variants" (list of strings)
                batch_ids = {item["oare_id"] for item in batch_items}
                results = {}
                for item in parsed:
                    if isinstance(item, dict) and "id" in item:
                        variants = item.get("variants", [])
                        if isinstance(variants, str):
                            variants = [variants]
                        results[item["id"]] = [v.strip() for v in variants if isinstance(v, str) and v.strip()]
                
                missing_ids = batch_ids - set(results.keys())
                if missing_ids:
                    raise IncompleteBatchError(f"Missing back-translations for {len(missing_ids)} items.")
                
                return results
                
            except json.JSONDecodeError:
                print(f"JSON error (intento {attempt+1}/{retries})")
                await asyncio.sleep(3)
            except IncompleteBatchError as ie:
                print(f"Batch Incompleto (intento {attempt+1}/{retries}): {ie}")
                await asyncio.sleep(2)
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "quota" in err or "rate" in err:
                    wait = 60
                    print(f"Rate limit, esperando {wait}s...")
                else:
                    wait = 5
                    print(f"API error ({attempt+1}/{retries}): {str(e)[:80]}")
                await asyncio.sleep(wait)
        
        # FALLBACK: intentar uno por uno
        print(f"Fallback: back-traduciendo individualmente...")
        fallback_results = locals().get('results', {})
        missing = [item for item in batch_items if item["oare_id"] not in fallback_results]
        
        for item in missing:
            try:
                single_prompt = build_batch_prompt([item])
                resp = await model.generate_content_async(
                    single_prompt,
                    generation_config=genai.types.GenerationConfig(
                        response_mime_type="application/json",
                    )
                )
                await asyncio.sleep(1)
                
                parsed = json.loads(resp.text.strip())
                if isinstance(parsed, dict) and "id" in parsed:
                    variants = parsed.get("variants", [])
                    if isinstance(variants, str):
                        variants = [variants]
                    fallback_results[parsed["id"]] = [v.strip() for v in variants if isinstance(v, str) and v.strip()]
                elif isinstance(parsed, list) and len(parsed) > 0:
                    p = parsed[0]
                    variants = p.get("variants", [])
                    if isinstance(variants, str):
                        variants = [variants]
                    fallback_results[p.get("id", "")] = [v.strip() for v in variants if isinstance(v, str) and v.strip()]
            except Exception as e2:
                print(f"Fallback falló para {item['oare_id'][:20]}: {str(e2)[:50]}")
        return fallback_results


async def async_main():
    print("Cargando datos de train.csv...")
    train = pd.read_csv(TRAIN_FILE)
    
    print(f"Oraciones originales: {len(train)}")
    print(f"Variaciones por oración: {VARIATIONS}")
    print(f"Total esperado: ~{len(train) * VARIATIONS} pares nuevos")
    
    # Checkpoint
    if os.path.exists(OUTPUT_FILE):
        out_df = pd.read_csv(OUTPUT_FILE)
        done_ids = set(out_df['oare_id'].tolist())
        print(f"Checkpoint: {len(done_ids)} ya procesados")
    else:
        out_df = pd.DataFrame(columns=['oare_id', 'transliteration', 'translation'])
        done_ids = set()
    
    # Preparar items: cada fila del train con su traducción en inglés
    pending = []
    for _, row in train.iterrows():
        doc_id = str(row['oare_id'])
        if doc_id not in done_ids:
            english = str(row['translation']).strip()
            if len(english) > 10:  # Ignorar oraciones muy cortas
                pending.append({
                    'oare_id': doc_id,
                    'english': english,
                    'original_translit': str(row.get('transliteration', '')).strip()
                })
    
    total = len(pending)
    print(f"\n{total} oraciones por back-traducir")
    print(f"Batch: {BATCH_SIZE} | Paralelo: {MAX_CONCURRENT} | Teórico: {BATCH_SIZE*MAX_CONCURRENT} en vuelo")
    
    # Crear batches
    batches = [pending[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    buffer = []
    
    with tqdm(total=total, desc="Back-Traduciendo") as pbar:
        for wave_start in range(0, len(batches), MAX_CONCURRENT):
            wave = batches[wave_start:wave_start + MAX_CONCURRENT]
            
            tasks = [translate_batch(batch, semaphore) for batch in wave]
            wave_results = await asyncio.gather(*tasks)
            
            for batch, variants_dict in zip(wave, wave_results):
                for item in batch:
                    doc_id = item['oare_id']
                    variants = variants_dict.get(doc_id, [])
                    
                    if variants:
                        for vi, variant in enumerate(variants):
                            buffer.append({
                                'oare_id': f"{doc_id}_bt{vi+1}",
                                'transliteration': variant,
                                'translation': item['english']
                            })
                    else:
                        print(f"Missing: {doc_id[:30]}")
                
                pbar.update(len(batch))
            
            # Checkpoint
            if len(buffer) >= SAVE_EVERY:
                temp_df = pd.DataFrame(buffer)
                out_df = pd.concat([out_df, temp_df], ignore_index=True)
                out_df.to_csv(OUTPUT_FILE, index=False)
                done_ids.update(temp_df['oare_id'].tolist())
                buffer = []
                print(f"Checkpoint: {len(out_df)} total")
    
    # Guardar restante
    if buffer:
        temp_df = pd.DataFrame(buffer)
        out_df = pd.concat([out_df, temp_df], ignore_index=True)
        out_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nBACK-TRANSLATION COMPLETADA")
    print(f"Filas generadas: {len(out_df)} pares sintéticos")
    print(f"-> {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass
    asyncio.run(async_main())