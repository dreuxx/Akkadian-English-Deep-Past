import eflomal

def align_corpus(src_file, trg_file, output_prefix, model=3):
    """
    Align a parallel corpus in both forward and reverse directions.

    Args:
        src_file (str): Path to the source language file.
        trg_file (str): Path to the target language file.
        output_prefix (str): Prefix for output files.
        model (int): Model type (1, 2, or 3). Default is 3.
    """
    aligner = eflomal.Aligner()

    forward_file = f"{output_prefix}.fwd"
    reverse_file = f"{output_prefix}.rev"

    print(f"Aligning corpus: {src_file} -> {trg_file}")
    with open(src_file, 'r', encoding='utf-8') as src_data, \
         open(trg_file, 'r', encoding='utf-8') as trg_data:
        aligner.align(
            src_data, trg_data,
            links_filename_fwd=forward_file,
            links_filename_rev=reverse_file,
        )

    print(f"Alignment completed. Forward: {forward_file}, Reverse: {reverse_file}")

if __name__ == "__main__":
    # Example usage
    src_file = "data/input/train.akk"
    trg_file = "data/input/train.en"
    output_prefix = "data/output/akk-en"

    align_corpus(src_file, trg_file, output_prefix)
