import os
import gc
import shutil
import re
import math
import pandas as pd
import numpy as np
import sacrebleu
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)

from sentence_transformers import SentenceTransformer, util
import evaluate

class Config:
    MODEL_NAME = "google/byt5-base"
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    EPOCHS = 9
    LEARNING_RATE_NOISY = 1e-4
    LEARNING_RATE_GOLD = 5e-5  # legacy; Phase 2 usa LEARNING_RATE_PHASE2
    LEARNING_RATE_PHASE2 = 1e-5
    DROPOUT_RATE = 0.15
    PREFIX = "translate Akkadian to English: "
    # True = cargar modelo de _phase09 y entrenar desde Phase 1
    START_FROM_PHASE_1 = True

    try:
        from google.colab import drive
        drive.mount('/content/drive')
        OUTPUT_DIR = "/content/drive/MyDrive/byt5-akkadian-model"
        print("Guardado en la nube activado: Google Drive Montado.")
    except ImportError:
        OUTPUT_DIR = "./byt5-akkadian-model"
        print("Google Colab no detectado. Los modelos se guardarán de forma local/temporal.")

def warmup_steps_from_data(num_train, batch_size, grad_accum, num_epochs, warmup_ratio=0.1):
    """Warmup steps as fraction of total steps (data-dependent)."""
    if num_train <= 0:
        return 0
    total = max(1, (num_train // (batch_size * grad_accum)) * num_epochs)
    return max(0, min(int(total * warmup_ratio), total - 1))

def calc_eval_steps(n_train: int, batch_size: int, grad_accum: int, epochs: int, n_evals: int = 10) -> int:
    steps_per_epoch = math.ceil(n_train / (batch_size * grad_accum))
    total_steps = steps_per_epoch * epochs
    return max(50, total_steps // n_evals)

def calc_warmup_steps(n_train: int, batch_size: int, grad_accum: int, epochs: int, ratio: float = 0.06) -> int:
    steps_per_epoch = math.ceil(n_train / (batch_size * grad_accum))
    total_steps = steps_per_epoch * epochs
    return max(10, int(total_steps * ratio))

def make_compute_metrics(tokenizer):
    """BLEU + chrF++ y score = sqrt(BLEU * chrF++) (mismo criterio que facil/train.py)."""
    vocab_size = tokenizer.vocab_size

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds = np.where(preds < vocab_size, preds, tokenizer.pad_token_id)
        preds = np.where(preds >= 0, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score
        chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels], word_order=2).score
        geo_mean = (
            math.sqrt(max(bleu, 0) * max(chrf, 0)) if (bleu > 0 and chrf > 0) else 0.0
        )
        return {"bleu": bleu, "chrf": chrf, "score": geo_mean}

    return compute_metrics

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

seed_everything()

# ==========================================
# HELPER: limpiar dir antes de guardar + descongelar tras cargar
# Razón: save_model() NO borra archivos viejos → pesos LoRA de runs
# anteriores pueden quedarse mezclados con el modelo nuevo.
# Solución: borrar el directorio antes de guardar, y forzar
# requires_grad=True en todos los parámetros tras cargar.
# ==========================================
def clean_save(trainer, save_dir):
    """
    Guarda el modelo de forma segura:
    1. Guarda primero en /tmp (local, sin latencia de Drive)
    2. Borra el destino en Drive
    3. Copia de /tmp → Drive atómicamente
    Esto evita que rmtree+write en Drive dejen archivos vacíos
    por problemas de sincronización del filesystem de Colab.
    """
    import tempfile, shutil as _sh
    # Paso 1: guardar en /tmp local (rápido, sin Drive)
    tmp_dir = tempfile.mkdtemp(prefix="byt5_save_")
    try:
        trainer.save_model(tmp_dir)
        # Verificar que el guardado produjo archivos reales
        saved_files = os.listdir(tmp_dir)
        total_size  = sum(os.path.getsize(os.path.join(tmp_dir, f))
                         for f in saved_files if os.path.isfile(os.path.join(tmp_dir, f)))
        if total_size < 1_000:  # menos de 1KB = algo salió mal
            raise RuntimeError(f"save_model produjo archivos vacíos: {saved_files}")
        print(f"Guardado temporal OK ({total_size/1e6:.0f} MB, {len(saved_files)} archivos)")
        # Paso 2: borrar destino viejo en Drive
        _sh.rmtree(save_dir, ignore_errors=True)
        # Paso 3: copiar /tmp → Drive
        _sh.copytree(tmp_dir, save_dir)
        print(f"Copiado a {save_dir}")
    finally:
        _sh.rmtree(tmp_dir, ignore_errors=True)

def load_full_model(model_path):
    """Carga modelo y descongelante todos los parámetros (fix anti-LoRA residual)."""
    m = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    for param in m.parameters():
        param.requires_grad_(True)
    return m

# ==========================================
# PREPROCESSING FUNCTIONS (Competition Recommendations)
# ==========================================

DECIMAL_TO_FRACTION = [
    ('0.8333', '⅚'), ('0.6666', '⅔'), ('0.3333', '⅓'),
    ('0.1666', '⅙'), ('0.625', '⅝'), ('0.25', '¼'),
    ('0.75', '¾'), ('0.5', '½'),
]

SUBSCRIPT_MAP = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')

ROMAN_TO_INT = {
    'XII': '12', 'XI': '11', 'VIII': '8', 'VII': '7',
    'VI': '6', 'IV': '4', 'III': '3', 'II': '2',
    'IX': '9', 'X': '10', 'V': '5', 'I': '1',
}

def clean_transliteration(text):
    """Clean transliteration text per competition recommendations."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\((?:large )?break\)', '<gap>', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\d+ broken lines?\)', '<gap>', text, flags=re.IGNORECASE)
    text = text.replace('…', '<gap>').replace('...', '<gap>')
    text = re.sub(r'\[x\]', '<gap>', text)
    text = re.sub(r'(?<!\w)x(?!\w)', '<gap>', text)
    text = text.replace('<big_gap>', '<gap>')
    text = re.sub(r'(<gap>\s*-?\s*){2,}', '<gap> ', text)
    text = re.sub(r'\(d\)', '{d}', text)
    text = re.sub(r'\(ki\)', '{ki}', text)
    text = re.sub(r'\(TÚG\)', 'TÚG', text)
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.replace('KÙ.B.', 'KÙ.BABBAR')
    text = text.translate(SUBSCRIPT_MAP)
    for dec, frac in DECIMAL_TO_FRACTION:
        text = text.replace(dec, frac)
    text = re.sub(r'(\d+\.\d{4})\d+', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_translation(text):
    """Clean translation text per competition recommendations."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\((?:large )?break\)', '<gap>', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\d+ broken lines?\)', '<gap>', text, flags=re.IGNORECASE)
    text = text.replace('…', '<gap>').replace('...', '<gap>')
    text = re.sub(r'\[x\]', '<gap>', text)
    text = text.replace('<big_gap>', '<gap>')
    text = re.sub(r'(<gap>\s*-?\s*){2,}', '<gap> ', text)
    text = re.sub(r'\(d\)', '{d}', text)
    text = re.sub(r'\(ki\)', '{ki}', text)
    text = re.sub(r'\(TÚG\)', 'TÚG', text)
    for remove in [' fem.', ' sing.', ' pl.', ' plural', '(?)']:
        text = text.replace(remove, '')
    text = re.sub(r'<<\s*>>', '', text)
    text = re.sub(r'(?<!\<)< >(?!\>)', '', text)
    text = re.sub(r'(?<!\.)\.\.(?!\.)', '', text)
    text = re.sub(r'\bxx\b', '', text)
    text = re.sub(r'\bPN\b', '<gap>', text)
    text = text.replace('-gold', 'pašallum gold')
    text = text.replace('-tax', 'šadduātum tax')
    text = text.replace('-textiles', 'kutānum textiles')
    text = text.replace('1 / 12 (shekel)', '15 grains')
    text = text.replace('5 / 12 shekel', '⅓ shekel 15 grains')
    text = text.replace('5 11 / 12 shekels', '6 shekels less 15 grains')
    text = text.replace('7 / 12 shekel', '½ shekel 15 grains')
    text = re.sub(r'\b([a-zA-Z]+)\s*/\s*[a-zA-Z]+\b', r'\1', text)
    for dec, frac in DECIMAL_TO_FRACTION:
        text = text.replace(dec, frac)
    text = re.sub(r'(\d+\.\d{4})\d+', r'\1', text)
    for roman, num in ROMAN_TO_INT.items():
        text = re.sub(r'\bmonth\s+' + roman + r'\b', f'month {num}', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# DATA LOADING
# ==========================================

# --- KAGGLEHUB COLAB AUTO-DOWNLOADER ---
def resolve_path(kaggle_path, hub_handle=None):
    if os.path.exists(kaggle_path):
        return kaggle_path
    elif hub_handle is not None:
        try:
            import kagglehub
            print(f"Resolviendo {hub_handle} vía kagglehub (Colab/Local)...")
            return kagglehub.dataset_download(hub_handle)
        except Exception as e:
            print(f"Error descargando {hub_handle}: {e}")
    return kaggle_path

INPUT_DIR = resolve_path("/kaggle/input/competitions/deep-past-initiative-machine-translation", "giovannyrodrguez/datotrain110")

TRAIN_FILE = f"{INPUT_DIR}/train.csv"

GEMINI_DIR = resolve_path("/kaggle/input/datasets/giovannyrodrguez/traduciongemeni", "giovannyrodrguez/traduciongemeni")
GEMINI_FILE = f"{GEMINI_DIR}/gemini_pseudo_labeled_train.csv"

BACKTRANS_DIR = resolve_path("/kaggle/input/datasets/giovannyrodrguez/backtralente", "giovannyrodrguez/backtralente")
BACKTRANS_FILE = f"{BACKTRANS_DIR}/gemini_back_translated_train.csv"

ONOMASTICON_DIR = resolve_path("/kaggle/input/datasets/giovannyrodrguez/mas-disonario", "giovannyrodrguez/mas-disonario")
ONOMASTICON_FILE = f"{ONOMASTICON_DIR}/onomasticon.csv"

test_df = pd.read_csv(f"{INPUT_DIR}/test.csv")

# ──────────────────────────────────────────────────────────────────
# AUTO-RESOLVE: rutas de Drive con fallback a kagglehub
# Si el archivo NO existe en Drive se descarga via kagglehub y
# se copia a la ruta de Drive para que quede guardado para siempre.
# ──────────────────────────────────────────────────────────────────
def resolve_file(drive_path, kaggle_handle, filename):
    """
    1. Si el archivo ya existe en Drive → lo usa directamente.
    2. Si no existe → descarga con kagglehub → copia a Drive.
    3. Si kagglehub también falla → devuelve la ruta original
       (el código aguas abajo ya maneja el caso vacío).
    """
    if os.path.exists(drive_path):
        return drive_path
    print(f"{filename} no encontrado en Drive. Descargando via kagglehub...")
    try:
        import kagglehub, shutil as _sh
        dl_dir = kagglehub.dataset_download(kaggle_handle)
        # Buscar el archivo dentro del directorio descargado
        for root, _, files in os.walk(dl_dir):
            for f in files:
                if f == filename:
                    src_path = os.path.join(root, f)
                    os.makedirs(os.path.dirname(drive_path), exist_ok=True)
                    _sh.copy2(src_path, drive_path)
                    print(f"{filename} copiado a {drive_path}")
                    return drive_path
        print(f"{filename} no encontrado dentro del dataset {kaggle_handle}")
    except Exception as e:
        print(f"Error descargando {kaggle_handle}: {e}")
    return drive_path   # devuelve la ruta aunque no exista; código aguas abajo lo maneja

try:
    import google.colab
    _DRIVE_CDLI      = "/content/drive/MyDrive/NuevoAKKadian"
    _DRIVE_DATA      = "/content/drive/MyDrive/Dataakkadian"
    _DRIVE_OARE      = "/content/drive/MyDrive/akkorae"

    CDLI_10K_FILE    = resolve_file(f"{_DRIVE_CDLI}/cdli_10k_gemini_pseudo_labels.csv",
                                    "giovannyrodrguez/cdliakkadian",
                                    "cdli_10k_gemini_pseudo_labels.csv")
    CDLI_JD_FILE     = resolve_file(f"{_DRIVE_DATA}/train_cdli_jd_fixed.csv",
                                    "giovannyrodrguez/cdliakkadian",
                                    "train_cdli_jd_fixed.csv")
    JOINT_DROPOUT_FILE = resolve_file(f"{_DRIVE_DATA}/train_jd_fixed.csv",
                                    "giovannyrodrguez/backtralente",
                                    "train_jd_fixed.csv")
    PHASE09_JD_FILE  = resolve_file(f"{_DRIVE_DATA}/train_phase09_jd.csv",
                                    "giovannyrodrguez/backtralente",
                                    "train_phase09_jd.csv")
    OARE_PSEUDO_FILE = resolve_file(f"{_DRIVE_OARE}/akkadian_publish_pseudo_labels.csv",
                                    "giovannyrodrguez/akkorae",
                                    "akkadian_publish_pseudo_labels.csv")
    OARE_JD_FILE     = resolve_file(f"{_DRIVE_OARE}/train_Akk_jd_fixed.csv",
                                    "giovannyrodrguez/akkorae",
                                    "train_Akk_jd_fixed.csv")
    GEMINI_JD_FILE   = resolve_file(f"{_DRIVE_DATA}/train_Gemine_jd_fixed.csv",
                                    "giovannyrodrguez/backtralente",
                                    "train_Gemine_jd_fixed.csv")
    # Phase 1.05: CSV ya filtrado (analyze_pub_phase195_lengths); opcional JD (Joint_Dropout + fix_jd_ids)
    PUBLICATION_PHASE195_FILE = f"{_DRIVE_DATA}/publication_extracted_pairs_phase195.csv"
    PUBLICATION_JD_FILE = f"{_DRIVE_DATA}/train_pair_jd.csv"
    TRAIN_COMPLETE_TRAIN_FILE = f"{_DRIVE_DATA}/tc_train_grouped.csv"
    TRAIN_COMPLETE_VAL_FILE = f"{_DRIVE_DATA}/tc_val_grouped.csv"
except ImportError:
    CDLI_10K_FILE    = "/home/dreuxx/Documents/balosento/facil/cdli_10k_gemini_pseudo_labels.csv"
    CDLI_JD_FILE     = "/home/dreuxx/Documents/balosento/facil/train_cdli_jd_fixed.csv"
    JOINT_DROPOUT_FILE = "/home/dreuxx/Documents/balosento/facil/train_jd_fixed.csv"
    PHASE09_JD_FILE  = "/home/dreuxx/Documents/balosento/facil/Joint_Dropout-main/data/output/train_phase09_jd.csv"
    OARE_PSEUDO_FILE = "/home/dreuxx/Documents/balosento/facil/akkadian_publish_pseudo_labels.csv"
    OARE_JD_FILE     = "/home/dreuxx/Documents/balosento/facil/Joint_Dropout-main/data/output/train_Akk_jd_fixed.csv"
    GEMINI_JD_FILE   = "/home/dreuxx/Documents/balosento/facil/Joint_Dropout-main/data/output/train_Gemine_jd_fixed.csv"
    PUBLICATION_PHASE195_FILE = "/home/dreuxx/Documents/balosento/facil/publication_extracted_pairs_phase195.csv"
    PUBLICATION_JD_FILE = "/home/dreuxx/Documents/balosento/facil/Joint_Dropout-main/data/output/train_pair_jd.csv"
    TRAIN_COMPLETE_TRAIN_FILE = "/home/dreuxx/Documents/balosento/facil/tc_train_grouped.csv"
    TRAIN_COMPLETE_VAL_FILE = "/home/dreuxx/Documents/balosento/facil/tc_val_grouped.csv"

LEXICON_PATH = f"{INPUT_DIR}/OA_Lexicon_eBL.csv"
EBL_PATH = f"{INPUT_DIR}/eBL_Dictionary.csv"

norm_map = {}
def_map = {}

if os.path.exists(LEXICON_PATH):
    lex_df = pd.read_csv(LEXICON_PATH).dropna(subset=['form', 'norm'])
    norm_map = dict(zip(lex_df['form'].astype(str).str.strip(), lex_df['norm'].astype(str).str.strip()))

if os.path.exists(EBL_PATH):
    ebl_df = pd.read_csv(EBL_PATH).dropna(subset=['word', 'definition'])
    def_map = dict(zip(ebl_df['word'].astype(str).str.strip(), ebl_df['definition'].astype(str).str.strip()))

print("Loading Datasets...")
train_df = pd.read_csv(TRAIN_FILE)
print(f"Phase 2 (Gold) Data: {len(train_df)} docs")

if os.path.exists(GEMINI_FILE):
    gemini_df = pd.read_csv(GEMINI_FILE)
    if 'translation' in gemini_df.columns:
        gemini_df = gemini_df[gemini_df['translation'] != '<FAILED - RETRY>']
    print(f"Phase 1 (Gemini) Data: {len(gemini_df)} docs")
else:
    print(f"Could not find {GEMINI_FILE}. Make sure it is added in Kaggle.")
    gemini_df = pd.DataFrame(columns=train_df.columns)

if os.path.exists(BACKTRANS_FILE):
    bt_df = pd.read_csv(BACKTRANS_FILE)
    bt_df = bt_df.dropna(subset=['transliteration', 'translation'])
    print(f"Phase 0.5 (Back-Translation) Data: {len(bt_df)} docs")
else:
    print(f"Could not find {BACKTRANS_FILE}. Skipping Phase 0.5.")
    bt_df = pd.DataFrame(columns=train_df.columns)

print(f"Dictionaries Loaded -> OA Norms: {len(norm_map)}, eBL English Defs: {len(def_map)}")

# ==========================================
# BUILD DICTIONARY TRAINING PAIRS (Phase 0)
# ==========================================
dict_pairs = []

for form, norm in norm_map.items():
    eng_def = def_map.get(norm)
    if not eng_def:
        eng_def = def_map.get(form)
    if eng_def and len(str(eng_def)) > 1:
        short_def = str(eng_def).split('.')[0].replace('"', '').strip()
        if len(short_def) > 1:
            dict_pairs.append({'transliteration': str(form), 'translation': short_def})

for word, definition in def_map.items():
    if word not in norm_map:
        short_def = str(definition).split('.')[0].replace('"', '').strip()
        if len(short_def) > 1:
            dict_pairs.append({'transliteration': str(word), 'translation': short_def})

dict_df = pd.DataFrame(dict_pairs).drop_duplicates(subset=['transliteration'])
print(f"Phase 0 (Dictionary Vocab): {len(dict_df)} word pairs")

if os.path.exists(ONOMASTICON_FILE):
    ono_df = pd.read_csv(ONOMASTICON_FILE)
    ono_pairs = []
    for _, row in ono_df.iterrows():
        name = str(row['Name']).strip()
        spellings = str(row.get('Spellings_semicolon_separated', '')).strip()
        if not name or name == 'nan' or not spellings or spellings == 'nan':
            continue
        name = name.replace('<big_gap>', '<gap>')
        for sp in spellings.split(';'):
            sp = sp.strip().replace('<big_gap>', '<gap>')
            if sp and len(sp) > 1:
                ono_pairs.append({'transliteration': sp, 'translation': name})
    ono_df_pairs = pd.DataFrame(ono_pairs).drop_duplicates(subset=['transliteration'])
    dict_df = pd.concat([dict_df, ono_df_pairs], ignore_index=True).drop_duplicates(subset=['transliteration'])
    print(f"Onomasticon: {len(ono_pairs)} name-spelling pairs added")
    print(f"Phase 0 Total: {len(dict_df)} pairs (vocab + names)")
    del ono_df, ono_pairs, ono_df_pairs
else:
    print(f"Could not find {ONOMASTICON_FILE}. Skipping onomasticon.")

def simple_sentence_aligner(df):
    aligned_data = []
    for idx, row in df.iterrows():
        src = str(row['transliteration'])
        tgt = str(row['translation'])
        tgt_sents = [t.strip() for t in re.split(r'(?<=[.!?])\s+', tgt) if t.strip()]
        src_lines = [s.strip() for s in src.split('\n') if s.strip()]
        if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > 3 and len(t) > 3:
                    aligned_data.append({'transliteration': s, 'translation': t})
        else:
            aligned_data.append({'transliteration': src, 'translation': tgt})
    return pd.DataFrame(aligned_data)

bt_expanded = simple_sentence_aligner(bt_df)

cdli_10k_df = pd.read_csv(CDLI_10K_FILE) if os.path.exists(CDLI_10K_FILE) else pd.DataFrame(columns=['transliteration', 'translation'])
cdli_jd_df = pd.read_csv(CDLI_JD_FILE) if os.path.exists(CDLI_JD_FILE) else pd.DataFrame(columns=['transliteration', 'translation'])

for df_to_clean in [cdli_jd_df]:
    if not df_to_clean.empty:
        df_to_clean['transliteration'] = df_to_clean['transliteration'].apply(clean_transliteration)
        df_to_clean['translation'] = df_to_clean['translation'].apply(clean_translation)
        df_to_clean.dropna(subset=['transliteration', 'translation'], inplace=True)

del bt_df

for df in [dict_df, gemini_df, bt_expanded]:
    df['transliteration'] = df['transliteration'].apply(clean_transliteration)
    df['translation'] = df['translation'].apply(clean_translation)
    df.dropna(subset=['transliteration', 'translation'], inplace=True)

print(f"Expanded Phase 0.5 (Back-Trans): {len(bt_expanded)} sentences (cleaned)")
print(f"Phase 1 (Gemini): {len(gemini_df)} docs (will be expanded in Phase 1)")
print(f"Phase 2 (Gold) Base: {len(train_df)} docs (will be expanded later)")

# ==========================================
# TOKENIZATION
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

def preprocess_function(examples):
    inputs = [Config.PREFIX + str(ex) for ex in examples["transliteration"]]
    targets = [str(ex) for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=Config.MAX_LENGTH, truncation=True)
    labels = tokenizer(targets, max_length=Config.MAX_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# ==========================================
# MODEL LOADING & METRICS
# ==========================================
gc.collect()
torch.cuda.empty_cache()

model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
print(f"Model loaded from {Config.MODEL_NAME}")

metric = evaluate.load("chrf")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"chrf": result["score"]}

# (Phase -2 removed: MLM+DAE no longer in pipeline.)

# --- PHASE 0: VOCABULARY PRE-TRAINING (DICTIONARY) ---
print("\n=========================================")
print("STARTING PHASE 0: VOCABULARY (DICTIONARY)")
print("=========================================")

ds_p0 = Dataset.from_pandas(dict_df).train_test_split(test_size=0.1, seed=42)
tok_train_p0 = ds_p0["train"].map(preprocess_function, batched=True)
tok_val_p0 = ds_p0["test"].map(preprocess_function, batched=True)
num_dict = len(dict_df)
del ds_p0, dict_df

_warmup_p0 = warmup_steps_from_data(len(tok_train_p0), Config.BATCH_SIZE, 2, 10)
args_p0 = Seq2SeqTrainingArguments(
    output_dir=Config.OUTPUT_DIR + "_phase0",
    eval_strategy="epoch", save_strategy="no",
    learning_rate=2e-4, bf16=False, fp16=False,
    per_device_train_batch_size=Config.BATCH_SIZE, per_device_eval_batch_size=Config.BATCH_SIZE,
    gradient_accumulation_steps=2, weight_decay=0.01,
    label_smoothing_factor=0.1,
    warmup_steps=_warmup_p0,
    save_total_limit=1, num_train_epochs=10,
    predict_with_generate=False, logging_steps=10, report_to="none",
    lr_scheduler_type="cosine"
)

trainer_p0 = Seq2SeqTrainer(
    model=model, args=args_p0,
    train_dataset=tok_train_p0, eval_dataset=tok_val_p0,
    data_collator=data_collator, processing_class=tokenizer
)

if num_dict > 0:
    trainer_p0.train()
    print(f"Phase 0 complete. {num_dict} vocabulary words.")
else:
    print("Skipping Phase 0 (No dictionary pairs found).")

clean_save(trainer_p0, Config.OUTPUT_DIR + "_phase0")
del trainer_p0, model, tok_train_p0, tok_val_p0
gc.collect()
torch.cuda.empty_cache()
model = load_full_model(Config.OUTPUT_DIR + "_phase0")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
print("Phase 0 saved successfully.")

# --- PHASE 0.5: BACK-TRANSLATION ---
print("\n=========================================")
print("STARTING PHASE 0.5: BACK-TRANSLATION")
print("=========================================")

ds_p05 = Dataset.from_pandas(bt_expanded).train_test_split(test_size=0.1, seed=42)
tok_train_p05 = ds_p05["train"].map(preprocess_function, batched=True)
tok_val_p05 = ds_p05["test"].map(preprocess_function, batched=True)
num_bt = len(bt_expanded)
del ds_p05, bt_expanded

_warmup_p05 = warmup_steps_from_data(len(tok_train_p05), Config.BATCH_SIZE, 2, 10)
args_p05 = Seq2SeqTrainingArguments(
    output_dir=Config.OUTPUT_DIR + "_phase05",
    eval_strategy="epoch", save_strategy="no",
    learning_rate=1e-4, bf16=False, fp16=False,
    per_device_train_batch_size=Config.BATCH_SIZE, per_device_eval_batch_size=Config.BATCH_SIZE,
    gradient_accumulation_steps=2, weight_decay=0.01,
    label_smoothing_factor=0.1,
    warmup_steps=_warmup_p05,
    save_total_limit=1, num_train_epochs=10,
    predict_with_generate=False, logging_steps=10, report_to="none",
    lr_scheduler_type="cosine"
)

trainer_p05 = Seq2SeqTrainer(
    model=model, args=args_p05,
    train_dataset=tok_train_p05, eval_dataset=tok_val_p05,
    data_collator=data_collator, processing_class=tokenizer
)

if num_bt > 0:
    trainer_p05.train()
else:
    print("Skipping Phase 0.5 (No back-translation data found).")

clean_save(trainer_p05, Config.OUTPUT_DIR + "_phase05")
del trainer_p05, model, tok_train_p05, tok_val_p05
gc.collect()
torch.cuda.empty_cache()
model = load_full_model(Config.OUTPUT_DIR + "_phase05")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
print("Phase 0 directory preserved on Drive")

_pdf_df_cache = None  # no cache (Phase -2 removed); PDF loaded from HF below

# --- PHASE 0.9: PDF-EXTRACTED TRANSLATIONS (HuggingFace, train + validation nativos) ---
print("\n=========================================")
print("STARTING PHASE 0.9: PDF TRANSLATIONS")
print("=========================================")

try:
    from datasets import load_dataset as hf_load_dataset, concatenate_datasets
    hf_ds = hf_load_dataset("phucthaiv02/akkadian-translation")
    pdf_train = hf_ds["train"].to_pandas()
    pdf_val = hf_ds["validation"].to_pandas()

    def _ensure_columns(df):
        if "transliteration" not in df.columns:
            cols = df.columns.tolist()
            if len(cols) >= 2:
                df = df.rename(columns={cols[0]: "transliteration", cols[1]: "translation"})
        return df[["transliteration", "translation"]].dropna()

    pdf_train = _ensure_columns(pdf_train)
    pdf_val = _ensure_columns(pdf_val)
    pdf_train["transliteration"] = pdf_train["transliteration"].apply(clean_transliteration)
    pdf_train["translation"] = pdf_train["translation"].apply(clean_translation)
    pdf_val["transliteration"] = pdf_val["transliteration"].apply(clean_transliteration)
    pdf_val["translation"] = pdf_val["translation"].apply(clean_translation)
    pdf_train.dropna(subset=["transliteration", "translation"], inplace=True)
    pdf_val.dropna(subset=["transliteration", "translation"], inplace=True)
    print(f"Phase 0.9 (PDF Data): train {len(pdf_train)}, val {len(pdf_val)} (cleaned)")

    ds_p09 = DatasetDict({
        "train": Dataset.from_pandas(pdf_train),
        "test": Dataset.from_pandas(pdf_val),
    })

    if os.path.exists(PHASE09_JD_FILE):
        jd_09 = pd.read_csv(PHASE09_JD_FILE)
        jd_09["transliteration"] = jd_09["transliteration"].apply(clean_transliteration)
        jd_09["translation"] = jd_09["translation"].apply(clean_translation)
        jd_09.dropna(subset=["transliteration", "translation"], inplace=True)
        val_09_trans = set(pdf_val["translation"].astype(str).str.strip().str.lower())
        jd_09 = jd_09[~jd_09["translation"].apply(lambda t: str(t).strip().lower() in val_09_trans)]
        jd_09_ds = Dataset.from_pandas(jd_09.drop(columns=["id"], errors="ignore"))
        ds_p09["train"] = concatenate_datasets([ds_p09["train"], jd_09_ds])
        print(f"Phase 0.9 Augmented with JD. Train: {len(ds_p09['train'])}, Val: {len(ds_p09['test'])}")

    tok_train_p09 = ds_p09["train"].map(preprocess_function, batched=True)
    tok_val_p09 = ds_p09["test"].map(preprocess_function, batched=True)
    num_pdf = len(pdf_train) + len(pdf_val)
    del ds_p09, pdf_train, pdf_val

    _warmup_p09 = warmup_steps_from_data(len(tok_train_p09), Config.BATCH_SIZE, 2, 5)
    args_p09 = Seq2SeqTrainingArguments(
        output_dir=Config.OUTPUT_DIR + "_phase09",
        eval_strategy="epoch", save_strategy="no",
        learning_rate=1e-4, bf16=False, fp16=False,
        per_device_train_batch_size=Config.BATCH_SIZE, per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=2, weight_decay=0.01,
        label_smoothing_factor=0.1,
        warmup_steps=_warmup_p09,
        save_total_limit=1, num_train_epochs=5,
        predict_with_generate=False, logging_steps=100, report_to="none",
        lr_scheduler_type="cosine"
    )

    trainer_p09 = Seq2SeqTrainer(
        model=model, args=args_p09,
        train_dataset=tok_train_p09, eval_dataset=tok_val_p09,
        data_collator=data_collator, processing_class=tokenizer
    )

    if num_pdf > 0:
        trainer_p09.train()
        print(f"Phase 0.9 complete. {num_pdf} PDF pairs (train+val).")
    else:
        print("Skipping Phase 0.9 (No PDF data).")

    clean_save(trainer_p09, Config.OUTPUT_DIR + "_phase09")
    del trainer_p09, model, tok_train_p09, tok_val_p09
    gc.collect()
    torch.cuda.empty_cache()
    model = load_full_model(Config.OUTPUT_DIR + "_phase09")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    print("Phase 0.5 directory preserved on Drive")
except Exception as e:
    print(f"Phase 0.9 failed: {e}. Continuing with current model.")

# --- PHASE 1: PRE-TRAINING (GEMINI PSEUDO-LABELS) ---
print("\n=========================================")
print("STARTING PHASE 1: PRE-TRAINING (GEMINI)")
print("=========================================")

gemini_jd_augmented = None
if os.path.exists(GEMINI_JD_FILE):
    gemini_jd_augmented = pd.read_csv(GEMINI_JD_FILE)
    print(f"Found Gemini JD Data: {len(gemini_jd_augmented)} docs (will add to train split only)")
else:
    print(f"Could not find {GEMINI_JD_FILE}. Skipping Joint Dropout augmentation.")

p1_train_df, p1_val_df = train_test_split(gemini_df, test_size=0.1, random_state=42)

if gemini_jd_augmented is not None:
    gemini_jd_augmented['transliteration'] = gemini_jd_augmented['transliteration'].apply(clean_transliteration)
    gemini_jd_augmented['translation'] = gemini_jd_augmented['translation'].apply(clean_translation)
    gemini_jd_augmented.dropna(subset=['transliteration', 'translation'], inplace=True)

    jd_before = len(gemini_jd_augmented)

    if 'oare_id' in gemini_jd_augmented.columns and 'oare_id' in p1_val_df.columns:
        val_ids = set(p1_val_df["oare_id"])
        gemini_jd_augmented = gemini_jd_augmented[~gemini_jd_augmented['oare_id'].isin(val_ids)]
        jd_leaked = jd_before - len(gemini_jd_augmented)
        print(f"Filtered {jd_leaked} JD pairs by exact oare_id match (data leak prevention)")
else:
        val_translations = set(p1_val_df['translation'].astype(str).str.strip().str.lower())
        gemini_jd_augmented = gemini_jd_augmented[
            ~gemini_jd_augmented['translation'].apply(lambda t: str(t).strip().lower() in val_translations)
        ]
        jd_leaked = jd_before - len(gemini_jd_augmented)
        print(f"Filtered {jd_leaked} JD pairs whose translation appears in val (data leak prevention)")

p1_train_exp = simple_sentence_aligner(p1_train_df)
p1_val_exp = simple_sentence_aligner(p1_val_df)

for df in [p1_train_exp, p1_val_exp]:
    df['transliteration'] = df['transliteration'].apply(clean_transliteration)
    df['translation'] = df['translation'].apply(clean_translation)
    df.dropna(subset=['transliteration', 'translation'], inplace=True)

ds_p1_train = Dataset.from_pandas(p1_train_exp)
ds_p1_val = Dataset.from_pandas(p1_val_exp)

if gemini_jd_augmented is not None:
    jd_ds = Dataset.from_pandas(gemini_jd_augmented.drop(columns=['id', 'oare_id'], errors='ignore'))
    ds_p1_train = concatenate_datasets([ds_p1_train, jd_ds])
    print(f"Gemini JD data added to train split only. Train: {len(ds_p1_train)}, Val: {len(ds_p1_val)}")
    del gemini_jd_augmented, jd_ds

tok_train_p1 = ds_p1_train.map(preprocess_function, batched=True)
tok_val_p1 = ds_p1_val.map(preprocess_function, batched=True)
num_gemini = len(ds_p1_train)
del ds_p1_train, ds_p1_val, gemini_df

_warmup_p1 = warmup_steps_from_data(len(tok_train_p1), Config.BATCH_SIZE, 2, 5)
args_p1 = Seq2SeqTrainingArguments(
    output_dir=Config.OUTPUT_DIR + "_phase1",
    eval_strategy="epoch", save_strategy="no",
    learning_rate=1e-4, bf16=False, fp16=False,
    per_device_train_batch_size=Config.BATCH_SIZE, per_device_eval_batch_size=Config.BATCH_SIZE,
    gradient_accumulation_steps=2, weight_decay=0.01,
    label_smoothing_factor=0.1,
    warmup_steps=_warmup_p1,
    save_total_limit=1, num_train_epochs=5,
    predict_with_generate=False, logging_steps=10, report_to="none",
    lr_scheduler_type="cosine"
)

trainer_p1 = Seq2SeqTrainer(
    model=model, args=args_p1,
    train_dataset=tok_train_p1, eval_dataset=tok_val_p1,
    data_collator=data_collator, processing_class=tokenizer
)

if num_gemini > 0:
    trainer_p1.train()
else:
    print("Skipping Phase 1 (No Gemini data found).")

clean_save(trainer_p1, Config.OUTPUT_DIR + "_phase1")
del trainer_p1, model, tok_train_p1, tok_val_p1
gc.collect()
torch.cuda.empty_cache()
model = load_full_model(Config.OUTPUT_DIR + "_phase1")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
print("Phase 0.9 directory preserved on Drive")

# --- Save Model After Preliminary Phases ---
model.save_pretrained(Config.OUTPUT_DIR + "_noisy_pretrain")
tokenizer.save_pretrained(Config.OUTPUT_DIR + "_noisy_pretrain")
print(f"Preliminary noisy pre-training complete. Model saved to {Config.OUTPUT_DIR}_noisy_pretrain")

    

# --- PHASE 1.5: CDLI PRE-TRAINING (JOINT DROPOUT) ---
print("\n=========================================")
print("STARTING PHASE 1.5: CDLI (JOINT DROPOUT)")
print("=========================================")

if not cdli_10k_df.empty:
    p15_train_df, p15_val_df = train_test_split(cdli_10k_df, test_size=0.1, random_state=42)

    if not cdli_jd_df.empty:
        print(f"Found Joint Dropout Data for Phase 1.5: {len(cdli_jd_df)} docs (will add to train split only)")

        cdli_jd_df['transliteration'] = cdli_jd_df['transliteration'].apply(clean_transliteration)
        cdli_jd_df['translation'] = cdli_jd_df['translation'].apply(clean_translation)
        cdli_jd_df.dropna(subset=['transliteration', 'translation'], inplace=True)

        jd_before = len(cdli_jd_df)

        if 'id' in cdli_jd_df.columns and 'id' in p15_val_df.columns:
            val_ids = set(p15_val_df["id"])
            cdli_jd_df = cdli_jd_df[~cdli_jd_df['id'].isin(val_ids)]
            jd_leaked = jd_before - len(cdli_jd_df)
            print(f"Filtered {jd_leaked} JD pairs by exact id match (leak prevention)")
        else:
            val_translations = set(p15_val_df['translation'].astype(str).str.strip().str.lower())
            cdli_jd_df = cdli_jd_df[
                ~cdli_jd_df['translation'].apply(lambda t: str(t).strip().lower() in val_translations)
            ]
            jd_leaked = jd_before - len(cdli_jd_df)
            print(f"Filtered {jd_leaked} JD pairs whose translation appears in Phase 1.5 val (leak prevention)")

    p15_train_exp = p15_train_df[["transliteration", "translation"]].copy()
    p15_val_exp = p15_val_df[["transliteration", "translation"]].copy()

    for df in [p15_train_exp, p15_val_exp]:
        df['transliteration'] = df['transliteration'].apply(clean_transliteration)
        df['translation'] = df['translation'].apply(clean_translation)
        df.dropna(subset=['transliteration', 'translation'], inplace=True)

    ds_train_p15 = Dataset.from_pandas(p15_train_exp)
    ds_val_p15 = Dataset.from_pandas(p15_val_exp)

    if not cdli_jd_df.empty:
        jd_ds = Dataset.from_pandas(cdli_jd_df.drop(columns=['id', 'oare_id'], errors='ignore'))
        ds_train_p15 = concatenate_datasets([ds_train_p15, jd_ds])
        print(f"JD data added to Phase 1.5 train split. Total Train: {len(ds_train_p15)}")

    tok_train_p15 = ds_train_p15.map(preprocess_function, batched=True)
    tok_val_p15 = ds_val_p15.map(preprocess_function, batched=True)

    _warmup_p15 = warmup_steps_from_data(len(tok_train_p15), Config.BATCH_SIZE, 2, 5)
    args_p15 = Seq2SeqTrainingArguments(
        output_dir=Config.OUTPUT_DIR + "_phase1_5",
        eval_strategy="epoch", save_strategy="no",
        learning_rate=Config.LEARNING_RATE_NOISY, bf16=False, fp16=False,
        per_device_train_batch_size=Config.BATCH_SIZE, per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=2, weight_decay=0.01,
        label_smoothing_factor=0.1,
        warmup_steps=_warmup_p15,
        lr_scheduler_type="cosine",
        save_total_limit=2, num_train_epochs=5,
        predict_with_generate=False, logging_steps=10, report_to="none"
    )

    trainer_p15 = Seq2SeqTrainer(
        model=model, args=args_p15,
        train_dataset=tok_train_p15, eval_dataset=tok_val_p15,
        data_collator=data_collator, processing_class=tokenizer
    )

    trainer_p15.train()
    clean_save(trainer_p15, Config.OUTPUT_DIR + "_phase1_5")

    del trainer_p15, ds_train_p15, ds_val_p15, tok_train_p15, tok_val_p15, cdli_jd_df, cdli_10k_df
    gc.collect()
    torch.cuda.empty_cache()
    model = load_full_model(Config.OUTPUT_DIR + "_phase1_5")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    print("Phase 1.5 complete (CDLI + JD).")
else:
    print("Skipping Phase 1.5 (CDLI base dataset empty or not found).")

# --- PHASE 1.9: OARE PSEUDO-LABELS + JOINT DROPOUT ---
print("\n=========================================")
print("STARTING PHASE 1.9: OARE PSEUDO-LABELS")
print("=========================================")

oare_jd_augmented = None
if os.path.exists(OARE_JD_FILE):
    oare_jd_augmented = pd.read_csv(OARE_JD_FILE)
    print(f"Found Joint Dropout Data: {len(oare_jd_augmented)} docs (will add to train split only)")
else:
    print(f"Could not find {OARE_JD_FILE}. Skipping Joint Dropout augmentation.")

oare_df = pd.DataFrame()
if os.path.exists(OARE_PSEUDO_FILE):
    oare_df = pd.read_csv(OARE_PSEUDO_FILE)
    print(f"OARE raw: {len(oare_df)} filas | columnas: {oare_df.columns.tolist()}")
    print(f"NaN en transliteration: {oare_df['transliteration'].isna().sum() if 'transliteration' in oare_df.columns else 'N/A'}")
    print(f"NaN en translation:     {oare_df['translation'].isna().sum() if 'translation' in oare_df.columns else 'N/A'}")
    if 'transliteration' in oare_df.columns and 'translation' in oare_df.columns:
        if 'oare_id' in oare_df.columns:
            oare_df = oare_df[['oare_id', 'transliteration', 'translation']].dropna(subset=['transliteration', 'translation'])
        else:
            oare_df = oare_df[['transliteration', 'translation']].dropna()
        oare_df['transliteration'] = oare_df['transliteration'].apply(clean_transliteration)
        oare_df['translation'] = oare_df['translation'].apply(clean_translation)
        oare_df.dropna(subset=['transliteration', 'translation'], inplace=True)
        print(f"Loaded {len(oare_df)} OARE pseudo-label pairs")

if not oare_df.empty:
    # Si OARE tiene menos de 500 pares el split 90/10 produce un val inútil.
    if len(oare_df) < 500:
        print(f"OARE solo tiene {len(oare_df)} pares — usando split 99/1 para no desperdiciar datos")
        test_size_19 = 0.01
    else:
        test_size_19 = 0.1
    oare_train_df, oare_val_df = train_test_split(oare_df, test_size=test_size_19, random_state=42)

    # Filtrar JD con oare_val_df (anti-leak), como Phase 1.5 y Phase 2
    if oare_jd_augmented is not None:
        oare_jd_augmented['transliteration'] = oare_jd_augmented['transliteration'].apply(clean_transliteration)
        oare_jd_augmented['translation'] = oare_jd_augmented['translation'].apply(clean_translation)
        oare_jd_augmented.dropna(subset=['transliteration', 'translation'], inplace=True)
        jd_before = len(oare_jd_augmented)
        if 'oare_id' in oare_jd_augmented.columns and 'oare_id' in oare_val_df.columns:
            val_ids = set(oare_val_df["oare_id"])
            oare_jd_augmented = oare_jd_augmented[~oare_jd_augmented['oare_id'].isin(val_ids)]
            jd_leaked = jd_before - len(oare_jd_augmented)
            print(f"Filtered {jd_leaked} JD pairs by exact oare_id match (data leak prevention)")
        else:
            val_translations = set(oare_val_df['translation'].astype(str).str.strip().str.lower())
            oare_jd_augmented = oare_jd_augmented[
                ~oare_jd_augmented['translation'].apply(lambda t: str(t).strip().lower() in val_translations)
            ]
            jd_leaked = jd_before - len(oare_jd_augmented)
            print(f"Filtered {jd_leaked} JD pairs whose translation appears in val (data leak prevention)")

    oare_train_exp = oare_train_df[["transliteration", "translation"]].copy()
    oare_val_exp = oare_val_df[["transliteration", "translation"]].copy()
    for df in [oare_train_exp, oare_val_exp]:
        df['transliteration'] = df['transliteration'].apply(clean_transliteration)
        df['translation'] = df['translation'].apply(clean_translation)
        df.dropna(subset=['transliteration', 'translation'], inplace=True)
    ds_p19 = DatasetDict({"train": Dataset.from_pandas(oare_train_exp), "test": Dataset.from_pandas(oare_val_exp)})
    del oare_df, oare_train_df, oare_val_df, oare_train_exp, oare_val_exp

    if oare_jd_augmented is not None:
        jd_ds = Dataset.from_pandas(oare_jd_augmented.drop(columns=['id', 'oare_id'], errors='ignore'))
        ds_p19["train"] = concatenate_datasets([ds_p19["train"], jd_ds])
        print(f"JD data added to train split only. Train: {len(ds_p19['train'])}, Val: {len(ds_p19['test'])}")
        del oare_jd_augmented, jd_ds

    tok_train_p19 = ds_p19["train"].map(preprocess_function, batched=True)
    tok_val_p19 = ds_p19["test"].map(preprocess_function, batched=True)
    del ds_p19

    _warmup_p19 = warmup_steps_from_data(len(tok_train_p19), Config.BATCH_SIZE, 2, 5)
    args_p19 = Seq2SeqTrainingArguments(
        output_dir=Config.OUTPUT_DIR + "_phase1_9",
        eval_strategy="epoch", save_strategy="no",
        learning_rate=Config.LEARNING_RATE_NOISY, bf16=False, fp16=False,
        per_device_train_batch_size=Config.BATCH_SIZE, per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=2, weight_decay=0.01,
        label_smoothing_factor=0.1,
        warmup_steps=_warmup_p19,
        lr_scheduler_type="cosine",
        save_total_limit=2, num_train_epochs=5,
        predict_with_generate=False, logging_steps=10, report_to="none"
    )

    trainer_p19 = Seq2SeqTrainer(
        model=model, args=args_p19,
        train_dataset=tok_train_p19, eval_dataset=tok_val_p19,
        data_collator=data_collator, processing_class=tokenizer
    )

    trainer_p19.train()
    clean_save(trainer_p19, Config.OUTPUT_DIR + "_phase1_9")

    del trainer_p19, tok_train_p19, tok_val_p19
    gc.collect()
    torch.cuda.empty_cache()
    model = load_full_model(Config.OUTPUT_DIR + "_phase1_9")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    print("Phase 1.9 complete (OARE + JD).")
    print("Phase 1.5 directory preserved: byt5-akkadian-model_phase1_5")
else:
    print("Skipping Phase 1.9 (OARE data empty or not found).")


# --- PHASE 2: FINE-TUNING (HUMAN GOLD STANDARD) ---
print("\n=========================================")
print("STARTING PHASE 2: FINE-TUNING (GOLD)")
print("=========================================")

jd_augmented = None
if os.path.exists(JOINT_DROPOUT_FILE):
    jd_augmented = pd.read_csv(JOINT_DROPOUT_FILE)
    print(f"Found Joint Dropout Data: {len(jd_augmented)} docs (will add to train split only)")
else:
    print(f"Could not find {JOINT_DROPOUT_FILE}. Skipping Joint Dropout augmentation.")

p2_train_df, p2_val_df = train_test_split(train_df, test_size=0.1, random_state=42)
del train_df

if jd_augmented is not None:
    jd_augmented['transliteration'] = jd_augmented['transliteration'].apply(clean_transliteration)
    jd_augmented['translation'] = jd_augmented['translation'].apply(clean_translation)
    jd_augmented.dropna(subset=['transliteration', 'translation'], inplace=True)

    jd_before = len(jd_augmented)

    if 'oare_id' in jd_augmented.columns and 'oare_id' in p2_val_df.columns:
        val_ids = set(p2_val_df["oare_id"])
        jd_augmented = jd_augmented[~jd_augmented['oare_id'].isin(val_ids)]
        jd_leaked = jd_before - len(jd_augmented)
        print(f"Filtered {jd_leaked} JD pairs by exact oare_id match (data leak prevention)")
    else:
        val_translations = set(p2_val_df['translation'].astype(str).str.strip().str.lower())
        jd_augmented = jd_augmented[
            ~jd_augmented['translation'].apply(lambda t: str(t).strip().lower() in val_translations)
        ]
        jd_leaked = jd_before - len(jd_augmented)
        print(f"Filtered {jd_leaked} JD pairs whose translation appears in val (data leak prevention)")

p2_train_exp = simple_sentence_aligner(p2_train_df)
p2_val_exp = simple_sentence_aligner(p2_val_df)

for df in [p2_train_exp, p2_val_exp]:
    df['transliteration'] = df['transliteration'].apply(clean_transliteration)
    df['translation'] = df['translation'].apply(clean_translation)
    df.dropna(subset=['transliteration', 'translation'], inplace=True)

ds_train_p2 = Dataset.from_pandas(p2_train_exp)
ds_val_p2 = Dataset.from_pandas(p2_val_exp)

if jd_augmented is not None:
    jd_ds = Dataset.from_pandas(jd_augmented.drop(columns=['id', 'oare_id'], errors='ignore'))
    ds_train_p2 = concatenate_datasets([ds_train_p2, jd_ds])
    print(f"JD data added to train split only. Train: {len(ds_train_p2)}, Val: {len(ds_val_p2)}")
    del jd_augmented, jd_ds

tok_train_p2 = ds_train_p2.map(preprocess_function, batched=True)
tok_val_p2 = ds_val_p2.map(preprocess_function, batched=True)
del ds_train_p2, ds_val_p2

p2_eval_steps = calc_eval_steps(len(tok_train_p2), Config.BATCH_SIZE, 2, Config.EPOCHS)
p2_warmup = calc_warmup_steps(len(tok_train_p2), Config.BATCH_SIZE, 2, Config.EPOCHS)
print(
    f"[Phase 2] eval_steps={p2_eval_steps}, warmup_steps={p2_warmup}, "
    f"lr={Config.LEARNING_RATE_PHASE2}; best checkpoint por eval_loss"
)

args_p2 = Seq2SeqTrainingArguments(
    output_dir=Config.OUTPUT_DIR + "_phase2",
    eval_strategy="steps", save_strategy="steps",
    eval_steps=p2_eval_steps, save_steps=p2_eval_steps,
    learning_rate=Config.LEARNING_RATE_PHASE2, bf16=False, fp16=False,
    per_device_train_batch_size=Config.BATCH_SIZE, per_device_eval_batch_size=Config.BATCH_SIZE,
    gradient_accumulation_steps=2, weight_decay=0.01,
    max_grad_norm=1.0,
    label_smoothing_factor=0.1,
    warmup_steps=p2_warmup,
    lr_scheduler_type="cosine",
    save_total_limit=2, num_train_epochs=Config.EPOCHS,
    predict_with_generate=True,
    generation_max_length=Config.MAX_LENGTH,
    logging_steps=10, report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer_p2 = Seq2SeqTrainer(
    model=model, args=args_p2,
    train_dataset=tok_train_p2, eval_dataset=tok_val_p2,
    data_collator=data_collator, processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer_p2.train()

# --- Save Final Model ---
FINAL_DIR = Config.OUTPUT_DIR + "_final"
trainer_p2.save_model(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
print("Phase 1 directory preserved. (Kept Phase 2 checkpoints for inspection)")
print(f"\n Training complete! Final model saved to {FINAL_DIR}")