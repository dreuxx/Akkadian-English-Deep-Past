#!/usr/bin/env python3
"""
Prepare Akkadian data for Joint Dropout pipeline.
Splits train.csv into separate source (transliteration) and target (translation) files.
"""
import os
import re
from datasets import load_dataset

OUTPUT_DIR = "data/input"
SRC_FILE = os.path.join(OUTPUT_DIR, "train.akk")
TRG_FILE = os.path.join(OUTPUT_DIR, "train.en")

def clean_for_alignment(text):
    """Light cleaning to make text alignment-friendly."""
    text = str(text)
    text = text.replace('"', '').replace("'", "")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    
    csv_path = "/home/dreuxx/Documents/balosento/facil/publication_extracted_pairs_phase195.csv"
    print(f"Reading {csv_path}...")
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    # Asegurar que las columnas existen
    if 'transliteration' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'transliteration', df.columns[1]: 'translation'})

    n = 0
    with open(SRC_FILE, 'w', encoding='utf-8') as fsrc, \
         open(TRG_FILE, 'w', encoding='utf-8') as ftrg:
        
        for _, row in df.iterrows():
            src = str(row['transliteration']).strip()
            trg = str(row['translation']).strip()
            
            if len(src.split()) < 2 or len(trg.split()) < 2:
                continue
            
            fsrc.write(src + '\n')
            ftrg.write(trg + '\n')
            n += 1
    
    print(f"✅ Preparados {n} pares de oraciones")
    print(f"   Source: {SRC_FILE}")
    print(f"   Target: {TRG_FILE}")

if __name__ == "__main__":
    main()
