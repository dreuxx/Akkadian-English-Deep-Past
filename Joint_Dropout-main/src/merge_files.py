def merge_files(src_file, trg_file, output_file):
    """
    Merge source and target files into a single file.

    Args:
        src_file (str): Path to the source language file.
        trg_file (str): Path to the target language file.
        output_file (str): Path to the merged output file.
    """
    with open(src_file, 'r', encoding='utf-8') as src, \
         open(trg_file, 'r', encoding='utf-8') as trg, \
         open(output_file, 'w', encoding='utf-8') as out:
        for src_line, trg_line in zip(src, trg):
            src_line, trg_line = src_line.strip(), trg_line.strip()
            if src_line and trg_line:
                out.write(f"{src_line} ||| {trg_line}\n")
    print(f"Merged file created: {output_file}")

if __name__ == "__main__":
    # Example usage
    src_file = "data/input/train.akk"
    trg_file = "data/input/train.en"
    output_file = "data/output/akk-en.merged"

    merge_files(src_file, trg_file, output_file)
