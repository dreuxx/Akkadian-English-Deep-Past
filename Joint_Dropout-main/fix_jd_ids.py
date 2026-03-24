import pandas as pd
import os

def fix_jd_ids(csv_input, src_txt, trg_txt, output_csv):
    # 1. Read the original CSV to get IDs
    print(f"Reading {csv_input}...")
    df_orig = pd.read_csv(csv_input)
    
    # 2. Replicate the filtering logic from prepare_data.py
    valid_ids = []
    for idx, row in df_orig.iterrows():
        src_line = str(row['transliteration']).strip()
        trg_line = str(row['translation']).strip()
        
        # Exact same filter as prepare_data.py
        if len(src_line.split()) < 2 or len(trg_line.split()) < 2:
            continue
        
        key = "oare_id" if "oare_id" in df_orig.columns else "id"
        valid_ids.append(row[key])
    
    # 3. Read the JD output files
    print(f"Reading JD outputs: {src_txt} and {trg_txt}...")
    with open(src_txt, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f]
    with open(trg_txt, 'r', encoding='utf-8') as f:
        trg_lines = [line.strip() for line in f]
        
    # Validation
    if len(valid_ids) != len(src_lines):
        print(f"WARNING: ID count ({len(valid_ids)}) does not match sentence count ({len(src_lines)})!")
        # If they don't match, we might have a problem. 
        # But prepare_data.py processes all lines that pass the filter.
    
    # 4. Create the final CSV
    # Ensure we only use as many lines as we have IDs (or vice versa)
    min_len = min(len(valid_ids), len(src_lines))
    
    id_name = "oare_id" if "oare_id" in df_orig.columns else "id"
    df_out = pd.DataFrame({
        id_name: valid_ids[:min_len],
        'transliteration': src_lines[:min_len],
        'translation': trg_lines[:min_len]
    })
    
    df_out.to_csv(output_csv, index=False)
    print(f"✅ Created FIXED CSV with {len(df_out)} rows: {output_csv}")

if __name__ == "__main__":
    fix_jd_ids(
        "/home/dreuxx/Documents/balosento/facil/publication_extracted_pairs_phase195.csv",
        "data/output/output_src.txt",
        "data/output/output_trg.txt",
        "data/output/train_pair.csv"
    )
