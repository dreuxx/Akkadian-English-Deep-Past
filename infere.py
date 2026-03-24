import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

MODEL_PATH = "/kaggle/input/modelofinalbyt5/byt5-akkadian-model_final"

TEST_DATA_PATH = "/kaggle/input/competitions/deep-past-initiative-machine-translation/test.csv"
BATCH_SIZE = 16
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        self.texts = df['transliteration'].astype(str).tolist()
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
        all_predictions.extend([d.strip() for d in decoded])

# --- Submission ---
submission = pd.DataFrame({
    "id": test_df["id"],
    "translation": all_predictions
})

submission["translation"] = submission["translation"].apply(lambda x: x if len(x) > 0 else "broken text")

submission.to_csv("submission.csv", index=False)
print("Submission file saved successfully!")
submission.head()



submission.to_csv("submission.csv", index=False)
print("Submission file saved successfully!")
submission.head()
