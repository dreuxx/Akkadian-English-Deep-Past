import json
import random
import re
from subword_nmt.apply_bpe import BPE
from nltk.tree import Tree
from sacremoses import MosesTokenizer, MosesDetokenizer

# Constants
VARIABLE_SET_SIZE = 10  # Number of variables for substitution
BPE_DIFF_THRESHOLD = 2  # Maximum acceptable BPE length difference
VARIABLE_PLACEHOLDER = "@@@@@"
MAX_PHRASES_PER_SENTENCE = 10  # Maximum phrases to replace in a sentence
NO_ALIGNMENT = -1  # Represents no alignment for a word
NEGATIVE_INDEX = -1  # Used as an invalid index or initial placeholder
NO_PHRASE_END = 0  # Used to check for an invalid end boundary

random.seed(12)

def load_config(config_path):
    """
    Load configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, "r") as f:
        return json.load(f)

def load_file(file_path):
    """
    Load a text file into a list of strings.

    Args:
        file_path (str): Path to the file.

    Returns:
        list[str]: Lines from the file as a list of strings.
    """
    with open(file_path, "r") as f:
        return [line.strip() for line in f]

def save_file(data, file_path):
    """
    Save a list of strings to a file.

    Args:
        data (list[str]): List of strings to save.
        file_path (str): Path to the output file.
    """
    with open(file_path, "w") as f:
        for line in data:
            f.write(line + "\n")

def extract_bpe_vocab(bpe_vocab_file, max_vocab):
    """
    Extract the most frequent BPE tokens from a vocabulary file.

    Args:
        bpe_vocab_file (str): Path to the BPE vocabulary file.
        max_vocab (int): Number of most frequent tokens to extract.

    Returns:
        list[str]: List of frequent BPE tokens.
    """
    with open(bpe_vocab_file, "r") as f:
        return [line.split()[0] for line in f.readlines()[:max_vocab]]

def parse_alignments(alignment_line):
    """
    Parse an alignment string into a list of tuples.

    Args:
        alignment_line (str): Alignment string in "src-tgt" format.

    Returns:
        list[tuple[int, int]]: List of alignment pairs.
    """
    return [[int(ali.split("-")[0]), int(ali.split("-")[1])] for ali in alignment_line.strip().split()]

def check_bpe_frequency(phrase, frequent_bpe_vocab):
    """
    Check if all tokens in a phrase are in the frequent BPE vocabulary.

    Args:
        phrase (str): Phrase to check.
        frequent_bpe_vocab (list[str]): List of frequent BPE tokens.

    Returns:
        bool: True if all tokens are in the frequent BPE vocabulary, False otherwise.
    """
    tokens = phrase.split()  # Split phrase into tokens
    return all(token in frequent_bpe_vocab for token in tokens)

def phrase_extraction(srctext, trgtext, alignment):
    """
    Extract phrase pairs from aligned source and target texts.

    Args:
        srctext (str): Source sentence.
        trgtext (str): Target sentence.
        alignment (list[tuple[int, int]]): Word alignment between source and target.

    Returns:
        set[tuple]: Set of extracted phrase pairs.
    """
    def extract(f_start, f_end, e_start, e_end):
        if f_end == NEGATIVE_INDEX:  # If no valid end is found
            return set()
        for e, f in alignment:
            if (f_start <= f <= f_end) and (e < e_start or e > e_end):
                return set()
        phrases = set()
        fs = f_start
        while True:
            fe = f_end
            while True:
                src_phrase = " ".join(srctext[i] for i in range(e_start, e_end + 1))
                trg_phrase = " ".join(trgtext[i] for i in range(fs, fe + 1))
                phrases.add(((e_start, e_end + 1), src_phrase, trg_phrase))
                fe += 1
                if fe in f_aligned or fe == len(trgtext):
                    break
            fs -= 1
            if fs in f_aligned or fs == NEGATIVE_INDEX:
                break
        return phrases

    srctext = srctext.split()
    trgtext = trgtext.split()
    srclen = len(srctext)
    trglen = len(trgtext)
    e_aligned = [i for i, _ in alignment]
    f_aligned = [j for _, j in alignment]
    bp = set()

    for e_start in range(srclen):
        for e_end in range(e_start, srclen):
            f_start, f_end = trglen - 1, NEGATIVE_INDEX
            for e, f in alignment:
                if e_start <= e <= e_end:
                    f_start = min(f, f_start)
                    f_end = max(f, f_end)
            phrases = extract(f_start, f_end, e_start, e_end)
            if phrases:
                bp.update(phrases)

    return bp

def sent_substitute(phrases, srctext, trgtext, length, vardrop, alignment, frequent_bpe_vocab):
    """
    Substitute phrases in the sentences with variables.

    Args:
        phrases (set[tuple]): Extracted phrase pairs.
        srctext (str): Source sentence.
        trgtext (str): Target sentence.
        length (int): Phrase length criterion.
        vardrop (float): Dropout rate for phrase substitution.
        alignment (list[tuple[int, int]]): Word alignment between source and target.
        frequent_bpe_vocab (list[str]): Frequent BPE tokens.

    Returns:
        tuple[str, str]: Modified source and target sentences.
    """
    def check_free_distance(sentence, phrase, order):
        variables = [f"X_{i}" for i in range(VARIABLE_SET_SIZE)] + [f"Y_{i}" for i in range(VARIABLE_SET_SIZE)]
        sentence = sentence.replace(phrase, VARIABLE_PLACEHOLDER, order)
        parts = sentence.split(VARIABLE_PLACEHOLDER)
        if len(parts[0].strip()) > NO_PHRASE_END and len(parts[1].strip()) > NO_PHRASE_END:
            left, right = parts[0].strip()[-4:], parts[1].strip()[:4]
            return left not in variables and right not in variables
        return True

    selected_phrase_indices = []
    to_be_replaced_src, to_be_replaced_tgt = [], []

    for phrase in sorted(list(phrases)):
        if abs(len(phrase[2].split()) - len(phrase[1].split())) < BPE_DIFF_THRESHOLD:
            src_phrase, trg_phrase = phrase[1], phrase[2]
            # Apply BPE frequency check
            if src_phrase in srctext and trg_phrase in trgtext:
                if check_bpe_frequency(src_phrase, frequent_bpe_vocab) and check_bpe_frequency(trg_phrase, frequent_bpe_vocab):
                    check_src = check_free_distance(srctext, src_phrase, 1)
                    check_tgt = check_free_distance(trgtext, trg_phrase, 1)
                    if check_src and check_tgt:
                        to_be_replaced_src.append([src_phrase, phrase[0]])
                        to_be_replaced_tgt.append([trg_phrase, phrase[0]])

    selected_phrase_indices = random.sample(range(len(to_be_replaced_src)), round(vardrop * len(to_be_replaced_src)))
    vars_src = [f"X_{i}" for i in range(len(selected_phrase_indices))]
    vars_trg = [f"Y_{i}" for i in range(len(selected_phrase_indices))]

    new_src_text = replace_with_var(srctext, selected_phrase_indices, to_be_replaced_src, vars_src)
    new_trg_text = replace_with_var(trgtext, selected_phrase_indices, to_be_replaced_tgt, vars_trg)

    return new_src_text, new_trg_text

def replace_with_var(original_text, selected_indices, to_be_replaced, vars):
    """
    Replace phrases in the text with corresponding variables.

    Args:
        original_text (str): Original sentence.
        selected_indices (list[int]): Indices of phrases to replace.
        to_be_replaced (list[tuple]): List of phrases to be replaced.
        vars (list[str]): Variables to use for replacement.

    Returns:
        str: Modified sentence.
    """
    intervals = [to_be_replaced[i][1] for i in selected_indices]
    text_split = original_text.split()
    new_text = ""
    for i, word in enumerate(text_split):
        replaced = False
        for j, interval in enumerate(intervals):
            if interval[0] <= i < interval[1]:
                replaced = True
                break
        new_text += (vars[j] if replaced else word) + " "
    return new_text.strip()

def main(config):
    """
    Main pipeline to process source and target sentences using Joint Dropout.

    Args:
        config (dict): Configuration parameters.
    """
    src_sents = load_file(config["src_file"])
    trg_sents = load_file(config["trg_file"])
    alignments = load_file(config["alignments_file"])
    frequent_bpe_vocab = extract_bpe_vocab(config["bpe_vocab_file"], config["max_vocab"])

    src_output, trg_output = [], []
    for i, (src, trg, alignment_line) in enumerate(zip(src_sents, trg_sents, alignments)):
        alignment = parse_alignments(alignment_line)
        phrases = phrase_extraction(src, trg, alignment)
        src_text, trg_text = sent_substitute(phrases, src, trg, config["length"], config["vardrop"], alignment, frequent_bpe_vocab)
        src_output.append(src_text)
        trg_output.append(trg_text)

    save_file(src_output, config["output_src"])
    save_file(trg_output, config["output_trg"])

if __name__ == "__main__":
    config_path = "config.json"
    config = load_config(config_path)
    main(config)
