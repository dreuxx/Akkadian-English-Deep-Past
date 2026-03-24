import eflomal

def generate_priors(src_file, trg_file, fwd_file, rev_file, priors_file):
    """
    Generate priors from aligned files.

    Args:
        src_file (str): Path to the source language file.
        trg_file (str): Path to the target language file.
        fwd_file (str): Path to the forward alignment file.
        rev_file (str): Path to the reverse alignment file.
        priors_file (str): Output file to store priors.
    """
    with open(src_file, 'r', encoding='utf-8') as src_data, \
         open(trg_file, 'r', encoding='utf-8') as trg_data, \
         open(fwd_file, 'r', encoding='utf-8') as fwd_links, \
         open(rev_file, 'r', encoding='utf-8') as rev_links, \
         open(priors_file, 'w', encoding='utf-8') as priors_out:
        priors = eflomal.calculate_priors(src_data, trg_data, fwd_links, rev_links)
        eflomal.write_priors(priors_out, *priors)
    print(f"Priors generated: {priors_file}")

if __name__ == "__main__":
    # Example usage
    src_file = "data/input/train.akk"
    trg_file = "data/input/train.en"
    fwd_file = "data/output/akk-en.fwd"
    rev_file = "data/output/akk-en.rev"
    priors_file = "data/output/akk-en.priors"

    generate_priors(src_file, trg_file, fwd_file, rev_file, priors_file)
