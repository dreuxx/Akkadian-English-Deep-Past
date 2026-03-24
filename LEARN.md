# LEARN.md — Akkadian to English (ByT5 curriculum)

This file is a **learning and operations guide** for the training pipeline in this repository. It complements the project README and stays close to what `trainv4.py` actually runs.

## Goal

Train **google/byt5-base** for **Akkadian transliteration → English translation**, using a **multi-phase curriculum**: dictionary and noisy sources first, then competition gold data. Every training example is prefixed with:

`translate Akkadian to English: <transliteration>`

(Target is the English `translation` column.)

## Repository layout (essentials)

| Path | Role |
|------|------|
| `trainv4.py` | Full curriculum training (all phases). |
| `infere.py` | Batch inference on competition `test.csv`. |
| `gemini_pseudo_labeler.py` | Optional: build Gemini pseudo-label CSVs for Phase 1-style data. |
| `gemini_back_translator.py` | Optional: back-translation CSVs for Phase 0.5. |
| `Joint_Dropout-main/` | Joint Dropout tooling and **pre-built JD CSVs** under `data/output/` when you run that pipeline locally. |

## Environment

Use a **GPU** (CUDA) for reasonable runtimes. Typical Python stack:

- `torch`, `transformers`, `datasets`, `accelerate`
- `pandas`, `numpy`, `scikit-learn`
- `sentence-transformers`, `evaluate`, `sacrebleu`
- Optional: `kagglehub` (auto-download when Kaggle mount paths are missing)
- Optional on Colab: `google.colab` (Drive mount inside `Config`)

Create a venv, install the packages your run actually imports (`trainv4.py` will error clearly if something is missing).

## Where data comes from

### Kaggle / local mirrors

The script tries **filesystem paths first** (e.g. `/kaggle/input/...`), then **`kagglehub.dataset_download(...)`** using the handles configured in `trainv4.py`. You need either:

- A Kaggle notebook with competition and optional datasets attached, or  
- Local copies at the same relative layout, or  
- Successful `kagglehub` downloads.

Main competition path (resolved via `INPUT_DIR`):

- `train.csv`, `test.csv`, lexicon files under the Deep Past MT competition layout.

Optional satellite datasets (handles appear in code): Gemini pseudo-labels, back-translation CSV, onomasticon, CDLI / OARE / JD derivatives, etc. **If a file is missing, most phases skip or degrade gracefully** (check the console logs).

### Google Colab + Drive

When `google.colab` is available, the script mounts Drive and sets:

- `Config.OUTPUT_DIR` → e.g. `/content/drive/MyDrive/byt5-akkadian-model`
- Extra CSV locations under `MyDrive/NuevoAKKadian`, `MyDrive/Dataakkadian`, `MyDrive/akkorae`, with `resolve_file` + kagglehub fallback.

Locally (no Colab), outputs default to `./byt5-akkadian-model` unless you change `Config`.

### Paths reserved for grouped “train complete” splits

`trainv4.py` defines **Colab and local paths** for:

- `tc_train_grouped.csv`, `tc_val_grouped.csv`
- `publication_extracted_pairs_phase195.csv` (and related JD pair CSVs)

These are **placeholders for your Drive or local mirror**. They are **not** wired into the sequential phase loop in the current script; add a phase or notebook cell if you want training to consume them explicitly.

## Training phases (executed in order)

Rough flow; see `trainv4.py` for exact arguments.

1. **Phase 0 — Dictionary / vocab**  
   OA Lexicon + eBL definitions (+ optional onomasticon). Word/short-definition pairs.

2. **Phase 0.5 — Back-translation**  
   Sentence pairs from the back-translation CSV (expanded with `simple_sentence_aligner` where line/sentence counts match).

3. **Phase 0.9 — PDF / Hugging Face**  
   Loads `phucthaiv02/akkadian-translation` (train/validation). Optional augmentation with `PHASE09_JD_FILE` Joint Dropout rows (leak filter vs. val translations).

4. **Phase 1 — Gemini pseudo-labels**  
   Train/val split on Gemini CSV; optional `GEMINI_JD_FILE` augmentation on **train only** (leak filter vs. val).

5. **Phase 1.5 — CDLI + Joint Dropout**  
   CDLI 10k-style CSV + optional `CDLI_JD_FILE`; JD added to train only with id- or translation-based val leak checks.

6. **Phase 1.9 — OARE pseudo-labels + JD**  
   OARE publish pseudo CSV + optional `OARE_JD_FILE`; adaptive val split if OARE is small; same JD leak filtering pattern.

7. **Phase 2 — Gold fine-tuning**  
   Competition `train.csv`, 90/10 split, `simple_sentence_aligner`, optional `JOINT_DROPOUT_FILE` on train only.  
   - **Optimization**: `eval_strategy` / `save_strategy` = **steps** (`calc_eval_steps`), warmup via `calc_warmup_steps`.  
   - **Best checkpoint**: `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, `greater_is_better=False`.  
   - **Early stopping**: `EarlyStoppingCallback(early_stopping_patience=3)`.  
   - Learning rate: `Config.LEARNING_RATE_PHASE2` (default **1e-5**).  
   - Generation: `predict_with_generate=True`, `generation_max_length=Config.MAX_LENGTH`.

**Checkpoint directories** (suffixes on `Config.OUTPUT_DIR`): `_phase0`, `_phase05`, `_phase09`, `_phase1`, `_phase1_5`, `_phase1_9`, `_phase2`, plus `_noisy_pretrain` and `_final` after Phase 2.

Saving to Drive uses **`clean_save`**: write to `/tmp`, verify size, replace destination to avoid partial/corrupt Drive sync.

## Important config flags

- `Config.PREFIX`: must match inference and any external scripts.
- `Config.DROPOUT_RATE`: used where dropout is applied in the modeling path (see model construction if you customize).

## Inference (`infere.py`)

- Loads **`MODEL_PATH`**: prefers `.../byt5-akkadian-model_final`, else `.../byt5-akkadian-model_phase1_9` (Colab Drive paths in file; adjust for local).
- Uses the **same** `PREFIX` as training: `translate Akkadian to English: `.
- Reads competition `test.csv` from `INPUT_DIR` and writes predictions for submission-style use.

For **local** runs, set `MODEL_PATH` to your saved `_final` folder (or symlink/copy).

## Joint Dropout (JD)

**Idea:** augments training with extra Akkadian–English pairs produced by the Joint Dropout pipeline; **always merged into train only**, with filters so JD rows do not duplicate validation translations (or ids when present).

CSV filenames expected by training include variants like `train_jd_fixed.csv`, phase-specific `*_jd_fixed.csv`, etc.  
The **`Joint_Dropout-main`** tree holds the research code and, after you run it, typical outputs under `data/output/`. Align those paths with the constants in `trainv4.py` (Colab vs local block).

## Data hygiene (shared with competition)

`clean_transliteration` and `clean_translation` normalize breaks, gaps, subscripts, some fractions, and competition-specific gloss quirks. **Inference MUST use the same cleaning** as training (`infere.py` mirrors transliteration cleaning).

## Troubleshooting

| Symptom | What to check |
|--------|----------------|
| Phase skipped immediately | Missing CSV; see log for path. Add data or kagglehub dataset. |
| CUDA OOM | Lower `Config.BATCH_SIZE` or gradient accumulation in `Seq2SeqTrainingArguments`. |
| Stale weights after rerun | `clean_save` and `load_full_model` exist to avoid mixed checkpoints; ensure you are loading the intended `OUTPUT_DIR_*` folder. |
| Bad leaderboard vs. val | Different distribution, beam/generate settings, or prefix mismatch vs. training. |

## Publishing to Git

This folder may be its own git root. If `git push` fails with authentication in a headless environment, run push from your machine with credentials or a personal access token.

---

**Summary:** Run `trainv4.py` with GPU + data paths resolved; Phase 2 produces `_final` for `infere.py`. Use this document as the map from **files on disk → phase → checkpoint folder**.
