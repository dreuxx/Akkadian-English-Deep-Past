import pandas as pd

def txt_to_csv(src_file, trg_file, out_csv):
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = f.read().splitlines()
    
    with open(trg_file, 'r', encoding='utf-8') as f:
        trg_lines = f.read().splitlines()
        
    df = pd.DataFrame({
        'id': [f"jd_{i}" for i in range(len(src_lines))],
        'transliteration': src_lines,
        'translation': trg_lines
    })
    
    df.to_csv(out_csv, index=False)
    print(f"✅ Created {out_csv} with {len(df)} sentence pairs.")

if __name__ == "__main__":
    txt_to_csv(
        "data/output/output_src.txt",
        "data/output/output_trg.txt",
        "data/output/train_Akk_jd.csv"
    )
