

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

# Cambiado a dataset de textos publicados
PUBLISHED_FILE = "/kaggle/input/datasets/giovannyrodrguez/public-txt/akkadian_publish_oare.csv"
TRAIN_FILE = "/kaggle/input/competitions/deep-past-initiative-machine-translation/train.csv"
LEXICON_FILE = "/kaggle/input/competitions/deep-past-initiative-machine-translation/OA_Lexicon_eBL.csv"
OUTPUT_FILE = "./akkadian_publish_pseudo_labels.csv"

# ============================================================
# PARÁMETROS DE PARALELISMO
# ============================================================
BATCH_SIZE = 15        # Chunks por llamada API
MAX_CONCURRENT = 10    # Llamadas API simultáneas (5*15=75 chunks en vuelo)
SAVE_EVERY = 150        # Guardar checkpoint cada N traducciones

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
# 3. LIMPIEZA SEGÚN INSTRUCCIONES OFICIALES
# ============================================================
def clean_transliteration(text: str) -> str:
    """
    Limpieza según INSTRUCCIONES OFICIALES del dataset.
    
    REMOVER: ! ? / : ˹ ˺ [ ] < > (mantener contenido interno)
    REEMPLAZAR: [x]→<gap>  …→<big_gap>  Ḫḫ→Hh
    PRESERVAR: {} (determinativos), <gap>, <big_gap>, . (Sumerogramas)
    """
    if not isinstance(text, str):
        return ""
    
    # Proteger marcadores válidos
    text = text.replace('<gap>', '§GAP§').replace('<big_gap>', '§BIGGAP§')
    
    # Remover notaciones modernas
    for ch in ['!', '?', '/', '˹', '˺']:
        text = text.replace(ch, '')
    text = text.replace(':', ' ')          # word divider → espacio
    text = text.replace('[', '').replace(']', '')  # mantener contenido
    text = text.replace('<', '').replace('>', '')   # scribal insertions
    
    # Reemplazar gaps
    text = re.sub(r'\bx\b', '§GAP§', text)
    text = text.replace('…', '§BIGGAP§')
    
    # Ḫ ḫ → H h (test data usa H h)
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    
    # Restaurar marcadores
    text = text.replace('§GAP§', '<gap>').replace('§BIGGAP§', '<big_gap>')
    
    return re.sub(r'\s+', ' ', text).strip()


# ============================================================
# 4. SPLIT SEMÁNTICO (REMOVIDO - SE DELEGA AL EXTRACTOR)
# ============================================================
# Los datos ya vienen fragmentados en <8 líneas por parte de cdli_extractor.py


# ============================================================
# 5. FEW-SHOT EXAMPLES (con h en vez de ḫ)
# ============================================================
FEW_SHOT_EXAMPLES = """
--- Example 1 (debt note, short) ---
Transliteration: 1 TÚG ša qá-tim i-tur₄-DINGIR il₅-qé
Translation: Itūr-ilī has received one textile of ordinary quality.

--- Example 2 (debt note with seals) ---
Transliteration: KIŠIB ma-nu-ba-lúm-a-šur DUMU ṣí-lá-{d}IM KIŠIB šu-{d}EN.LÍL DUMU ma-nu-ki-a-šur KIŠIB MAN-a-šur DUMU a-ta-a 0.3333 ma-na 2 GÍN KÙ.BABBAR SIG₅ i-ṣé-er PÚZUR₄-a-šur DUMU a-ta-a a-lá-hu-um i-šu iš-tù ha-muš-tim ša ì-lí-dan ITU.KAM ša ke-na-tim li-mu-um e-na-sú-in a-na ITU 14 ha-am-ša-tim i-ša-qal šu-ma lá iš-qú-ul 1.5 GÍN.TA a-na 1 ma-na-im i-na ITU.1.KAM ṣí-ib-tám ú-ṣa-áb
Translation: Seal of Mannum-balum-Aššur son of Ṣilli-Adad, seal of Šu-Illil son of Mannum-kī-Aššur, seal of Puzur-Aššur son of Ataya. Puzur-Aššur son of Ataya owes 22 shekels of good silver to Ali-ahum. Reckoned from the week of Ilī-dan, month of Ša-kēnātim, in the eponymy of Enna-Suen, he will pay in 14 weeks. If he has not paid in time, he will add interest at the rate 1.5 shekel per mina per month.

--- Example 3 (letter with gaps) ---
Transliteration: TÚG u-la i-dí-na-ku-um i-tù-ra-ma 9 GÍN KÙ.BABBAR
Translation: <gap> he did not give you a textile. He returned and 9 shekels of silver <gap>

--- Example 4 (letter) ---
Transliteration: um-ma šu-ku-tum-ma a-na IŠTAR-lá-ma-sí ù ni-ta-ah-šu-šar qí-bi₄-ma mì-šu ša ta-áš-pu-ra-ni-ni um-ma a-tí-na-ma É-tum a-na lá be-tim i-tù-ar a-pu-tum a-na en-um-a-šùr i-<gap>-ni-ma e ší-na ga <gap> ša lá ta-ha-dì-ri a-na IŠTAR-lá-ma-sí qí-bi₄-ma šu-ma a-ha-tí a-ta li-ba-am dì-ni-ší-im lá ta-ha-da-ar
Translation: From Šukkutum to Ištar-lamassī and Nitahšušar: Why is that you (fem. plur.) have written me, saying: The house is no longer a house." Urgent, to Ennam-Aššur <gap> Do not fear!. To Ištar-lamassī: If you are truly my sister, then encourage her. Do not fear.

--- Example 5 (inventory list) ---
Transliteration: 17 GÍN KÙ.BABBAR ší-im e-ma-ri-im 30 ma-na URUDU IŠTAR-pí-lá-ah a-na ar-be-e-šu <gap> ù-kà-pu-ú ša-pì-ú-tim 9 ma-na URUDU SIG₅ i ší-im sú-ub-ri-im 1 ma-na URUDU ší-kam iš-tí DUMU IŠTAR-ba-ni 2 kà-ar-pát kà-mu-nu
Translation: 17 shekels of silver, the price of a donkey, 30 minas of copper of Istar-pilah, 4 sets of thick saddlecloths, 9 minas of refined copper from the price of a slave, 1 mina of sikku-copper with the son of Istar-bani, 2 jars of cumin
"""


# ============================================================
# 6. PROMPT + API CALLS
# ============================================================
def build_batch_prompt(batch_items: list) -> str:
    docs = []
    for item in batch_items:
        doc = {"oare_id": item['id'], "transliteration": item['chunk']}
        if item.get('hints_str'):
            doc["dictionary_hints"] = item['hints_str']
        docs.append(doc)
    
    return f"""You are a world-class Assyriologist specializing in Old Assyrian (OA) Akkadian from Kanesh (Kültepe), ca. 1950-1850 BC.

TASK: Translate {len(batch_items)} cuneiform transliterations into fluent English.

CONVENTIONS:
1. Personal names → Akkadian reading: PÚZUR₄-a-šùr → "Puzur-Aššur"
2. KIŠIB → "Seal of". DUMU → "son of".
3. KÙ.BABBAR→"silver", KÙ.GI→"gold", URUDU→"copper", AN.NA→"tin", TÚG→"textile(s)", GÍN→"shekel(s)", ma-na→"mina(s)", GÚ→"talent(s)"
4. 0.3333→"1/3", 0.6666→"2/3", 0.5→"1/2"
5. <gap>→"<gap>", <big_gap>→"<big_gap>"
6. Letters: "From [sender] to [recipient]: ..."

EXAMPLES:
{FEW_SHOT_EXAMPLES}

INPUT:
{json.dumps(docs, indent=2, ensure_ascii=False)}

Return ONLY a JSON array. Each object: {{"oare_id": "...", "translation": "..."}}. No markdown. Translate ALL documents.

JSON:"""

class IncompleteBatchError(Exception):
    pass

async def translate_batch(batch_items: list, semaphore: asyncio.Semaphore, retries: int = 3) -> dict:
    """Traduce un batch con semáforo para controlar concurrencia."""
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
                await asyncio.sleep(0.5)  # cortesía mínima
                
                parsed = json.loads(response.text.strip())
                if isinstance(parsed, dict):
                    parsed = [parsed]
                
                # Check for missing ID translations
                batch_ids = {item["id"] for item in batch_items}
                results = {item.get("oare_id", ""): item.get("translation", "").strip()
                        for item in parsed
                        if isinstance(item, dict) and "oare_id" in item and "translation" in item}
                        
                missing_ids = batch_ids - set(results.keys())
                if missing_ids:
                    # Throw exception to force retry or fallback
                    raise IncompleteBatchError(f"Missing translations for {len(missing_ids)} items.")
                
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
        
        # FALLBACK: intentar uno por uno si el batch falló
        print(f"Fallback: traduciendo chunks missing individualmente...")
        # Note: 'results' might be unbound here if all attempts failed completely
        # so we default to an empty dict if it doesn't exist
        fallback_results = locals().get('results', {})
        missing = [item for item in batch_items if item["id"] not in fallback_results]
        
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
                if isinstance(parsed, dict) and "oare_id" in parsed:
                    fallback_results[parsed.get("oare_id", "")] = parsed.get("translation", "").strip()
                elif isinstance(parsed, dict) and "id" in parsed:
                    fallback_results[parsed.get("id", "")] = parsed.get("translation", "").strip()
                elif isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                    key = parsed[0].get("oare_id") or parsed[0].get("id", "")
                    fallback_results[key] = parsed[0].get("translation", "").strip()
            except Exception as e2:
                print(f"Fallback falló para ID {item['id'][:20]}: {str(e2)[:50]}")
        return fallback_results


async def async_main():
    print("Cargando datos...")
    pub = pd.read_csv(PUBLISHED_FILE)
    train = pd.read_csv(TRAIN_FILE)
    lexicon = load_lexicon(LEXICON_FILE)
    
    print(f"Diccionario: {len(lexicon)} entradas" if lexicon else "Sin diccionario")
    
    print(f"Published: {len(pub)}")
    # En este dataset usamos oare_id como único identificador
    to_translate_dedup = pub.drop_duplicates(subset=['transliteration'], keep='first')
    dup_count = len(pub) - len(to_translate_dedup)
    
    print(f"Duplicados por texto: {dup_count}")
    print(f"Únicos a traducir: {len(to_translate_dedup)}")
    
    # Checkpoint leyendo 'oare_id'
    if os.path.exists(OUTPUT_FILE):
        out_df = pd.read_csv(OUTPUT_FILE)
        done_ids = set(out_df['oare_id'].apply(str).tolist()) if 'oare_id' in out_df.columns else set()
        print(f"Checkpoint: {len(done_ids)} ya procesados")
    else:
        out_df = pd.DataFrame(columns=['oare_id', 'transliteration', 'translation'])
        done_ids = set()
    
    # Preparar chunks
    pending = []
    for _, row in to_translate_dedup.iterrows():
        clean_text = clean_transliteration(str(row['transliteration']))
        cid = str(row['oare_id'])
        if cid not in done_ids:
            hints = get_hints_for_chunk(clean_text, lexicon)
            pending.append({
                'id': cid, 
                'chunk': clean_text, 
                'hints_str': hints
            })
    
    total = len(pending)
    print(f"\n{total} chunks por traducir")
    print(f"Batch: {BATCH_SIZE} | Paralelo: {MAX_CONCURRENT} | Teórico: {BATCH_SIZE*MAX_CONCURRENT} chunks en vuelo")
    
    # Crear batches
    batches = [pending[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    buffer = []
    completed = 0
    
    with tqdm(total=total, desc="Traduciendo") as pbar:
        # Procesar en waves de MAX_CONCURRENT batches
        for wave_start in range(0, len(batches), MAX_CONCURRENT):
            wave = batches[wave_start:wave_start + MAX_CONCURRENT]
            
            # Lanzar TODOS los batches de esta wave en PARALELO
            tasks = [translate_batch(batch, semaphore) for batch in wave]
            wave_results = await asyncio.gather(*tasks)
            
            # Procesar resultados
            for batch, translations_dict in zip(wave, wave_results):
                for item in batch:
                    doc_id = item['id']
                    translation = translations_dict.get(doc_id)
                    buffer.append({
                        'oare_id': doc_id,
                        'transliteration': item['chunk'],
                        'translation': translation if translation else '<FAILED - RETRY>'
                    })
                    if not translation:
                        print(f"Missing: {doc_id[:30]}")
                
                completed += len(batch)
                pbar.update(len(batch))
            
            # Checkpoint
            if len(buffer) >= SAVE_EVERY:
                temp_df = pd.DataFrame(buffer)
                out_df = pd.concat([out_df, temp_df], ignore_index=True)
                out_df = out_df.drop_duplicates(subset=['oare_id'], keep='last')
                out_df.to_csv(OUTPUT_FILE, index=False)
                done_ids.update(temp_df['oare_id'].apply(str).tolist())
                buffer = []
                ok = len(out_df[out_df['translation'] != '<FAILED - RETRY>'])
                print(f"Checkpoint: {len(out_df)} total ({ok} OK)")
    
    # Guardar restante
    if buffer:
        temp_df = pd.DataFrame(buffer)
        out_df = pd.concat([out_df, temp_df], ignore_index=True)
        out_df = out_df.drop_duplicates(subset=['oare_id'], keep='last')
        out_df.to_csv(OUTPUT_FILE, index=False)
    
    # Dedup final: una sola fila por transliteración única
    before_dedup = len(out_df)
    out_df = out_df.drop_duplicates(subset=['transliteration'], keep='first')
    removed = before_dedup - len(out_df)
    if removed > 0:
        print(f"\nDedup por transliteración: {before_dedup} -> {len(out_df)} ({removed} repetidas eliminadas)")
        out_df.to_csv(OUTPUT_FILE, index=False)
    
    failed = (out_df['translation'] == '<FAILED - RETRY>').sum()
    print(f"\nCOMPLETADO")
    print(f"Filas: {len(out_df)} | OK: {len(out_df)-failed} | Fallidas: {failed}")
    print(f"-> {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass
    asyncio.run(async_main())

