import os
import re
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

# --- KAGGLEHUB COLAB AUTO-DOWNLOADER (igual a train) ---
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

# Mismo OUTPUT_DIR + "_final" que dpc_starter_train (Colab). Si no existe, usa phase1_9.
_FINAL = "/content/drive/MyDrive/byt5-akkadian-model_final"
_PHASE19 = "/content/drive/MyDrive/byt5-akkadian-model_phase1_9"
MODEL_PATH = _FINAL if os.path.isdir(_FINAL) else _PHASE19
print(f"MODEL_PATH = {MODEL_PATH}")
INPUT_DIR = resolve_path(
    "/kaggle/input/competitions/deep-past-initiative-machine-translation",
    "giovannyrodrguez/datotrain110",
)
TEST_DATA_PATH = f"{INPUT_DIR}/test.csv"
BATCH_SIZE = 16
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Cleaning (same as train clean_transliteration) ---
DECIMAL_TO_FRACTION = [
    ("0.8333", "⅚"), ("0.6666", "⅔"), ("0.3333", "⅓"),
    ("0.1666", "⅙"), ("0.625", "⅝"), ("0.25", "¼"),
    ("0.75", "¾"), ("0.5", "½"),
]
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def clean_transliteration(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\((?:large )?break\)", "<gap>", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\d+ broken lines?\)", "<gap>", text, flags=re.IGNORECASE)
    text = text.replace("…", "<gap>").replace("...", "<gap>")
    text = re.sub(r"\[x\]", "<gap>", text)
    text = re.sub(r"(?<!\w)x(?!\w)", "<gap>", text)
    text = text.replace("<big_gap>", "<gap>")
    text = re.sub(r"(<gap>\s*-?\s*){2,}", "<gap> ", text)
    text = re.sub(r"\(d\)", "{d}", text)
    text = re.sub(r"\(ki\)", "{ki}", text)
    text = re.sub(r"\(TÚG\)", "TÚG", text)
    text = text.replace("Ḫ", "H").replace("ḫ", "h")
    text = text.replace("KÙ.B.", "KÙ.BABBAR")
    text = text.translate(SUBSCRIPT_MAP)
    for dec, frac in DECIMAL_TO_FRACTION:
        text = text.replace(dec, frac)
    text = re.sub(r"(\d+\.\d{4})\d+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Model Loading ---
print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# --- Data Preparation ---
test_df = pd.read_csv(TEST_DATA_PATH)



PREFIX = "translate Akkadian to English: "

class InferenceDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df["transliteration"].astype(str).tolist()
        self.texts = [PREFIX + i for i in self.texts]
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text, 
            max_length=MAX_LENGTH, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }

test_dataset = InferenceDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Inference Loop ---
print("Starting Inference...")
all_predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
  
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_LENGTH,
            num_beams=4,
            early_stopping=True
        )
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # Remove null bytes/control artifacts occasionally emitted by broken generations.
        cleaned = [d.replace("\x00", "").strip() for d in decoded]
        all_predictions.extend(cleaned)

# --- Submission ---
submission = pd.DataFrame({
    "id": test_df["id"],
    "translation": all_predictions
})

submission["translation"] = submission["translation"].apply(lambda x: x if len(x) > 0 else "<empty>")

submission.to_csv("submission.csv", index=False)
print("Submission file saved successfully!")
submission.head()