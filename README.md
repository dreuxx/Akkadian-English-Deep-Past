# Akkadian-English Deep Past

Team members:
- Arnav Mishra ([Kaggle](https://www.kaggle.com/arnavmishra02))

Acknowledgement:
- Takamichi Toda ([Kaggle](https://www.kaggle.com/takamichitoda)) for the professional baseline that this solution builds on.

## Model

- Base model: `google/byt5-base`
- Input prefix: `translate Akkadian to English: `

### Why ByT5?

Akkadian transliterations contain many diacritics (`š, ṣ, ṭ, ḫ`), subscript numbers, and Unicode fractions (`⅓`, `⅔`). Subword tokenizers can struggle with this. ByT5 is byte-level, so it handles these symbols natively without vocabulary fragmentation.

## Datasets and Curriculum

We used 7 sources in increasing-quality curriculum order:

| Phase | Source | What it is |
|---|---|---|
| 0 | OA Lexicon + eBL Dictionary + Onomasticon | Word-level pairs (`form -> English definition`, name spellings -> canonical names) |
| 0.5 | Gemini back-translations | Synthetic sentence pairs from back-translating English into Akkadian |
| 0.9 | HuggingFace `phucthaiv02/akkadian-translation` | Translation pairs extracted from academic PDFs |
| 1 | Gemini pseudo-labels (OARE corpus) | LLM-generated translations of the full OARE transliteration corpus |
| 1.5 | CDLI 10K + Gemini pseudo-labels | 10K CDLI texts translated by Gemini |
| 1.9 | Published OARE translations | Translations from scholarly publications |
| 2 | Competition `train.csv` | Human gold-standard annotations |

All phases were augmented with Joint Dropout (JD): perturbed copies of training pairs where tokens are randomly dropped from both source and target. This helps with fragmentary tablets and missing spans. JD gave consistent LB gains (about `+0.2` per dataset), which compounds across phases.

Anti-leak filtering was applied in every phase: JD pairs matching validation IDs or validation translations were removed before training.

## Preprocessing

Mostly based on the host discussion post, plus project-specific fixes:

- Break/gap markers (`large break`, `N broken lines`, `…`, `[x]`, lone `x`) -> single `<gap>` token; consecutive gaps collapsed
- Determinatives preserved: `(d) -> {d}`, `(ki) -> {ki}`
- `Ḫ/ḫ -> H/h`, subscript digits -> regular digits
- Decimal fractions -> Unicode fractions (`0.3333 -> ⅓`)
- `KÙ.B. -> KÙ.BABBAR`
- Target-side replacements: `-gold -> pašallum gold`, `-tax -> šadduātum tax`, `-textiles -> kutānum textiles`
- Shekel fraction arithmetic normalization: `1/12 (shekel) -> 15 grains`, `5/12 shekel -> ⅓ shekel 15 grains`, etc.
- Roman month numerals -> Arabic (`month XII -> month 12`)
- `PN -> <gap>` and removal of linguistic noise tags (`fem.`, `sing.`, `pl.`, `(?)`)

A simple sentence aligner splits multi-sentence documents into 1:1 pairs when source line count matches target sentence count.

## Training

7-phase curriculum: each phase loads the previous checkpoint and continues training (noisy-to-clean progression).

| Phase | Epochs | LR | Key detail |
|---|---:|---:|---|
| 0 (Dictionary) | 10 | `2e-4` | Teaches word-level mappings |
| 0.5 (Back-trans) | 10 | `1e-4` | Synthetic fluency |
| 0.9 (PDF) | 5 | `1e-4` | Academic-style exposure |
| 1 (Gemini) | 5 | `1e-4` | Broad OARE coverage |
| 1.5 (CDLI) | 5 | `1e-4` | Broader cuneiform genres |
| 1.9 (Published) | 5 | `1e-4` | Scholarly-quality signal |
| 2 (Gold) | 9 | `1e-5` | Early stopping (`patience=3`), best checkpoint by `eval_loss` |

Common settings across phases:

- Batch size: `16`
- Gradient accumulation: `2` (effective batch size `32`)
- Label smoothing: `0.1`
- Weight decay: `0.01`
- LR scheduler: cosine

Phase 2 uses a lower LR (`1e-5`) to reduce catastrophic forgetting of pretraining knowledge. `EarlyStopping` + `load_best_model_at_end` ensures checkpoint selection by best validation loss.

## Validation

- Split: `90/10` random split (`seed=42`) per phase dataset
- Checkpoint selection in Phase 2: `eval_loss`
- Reported metrics: BLEU, chrF++, and geometric mean `sqrt(BLEU * chrF++)` = `LB = 37.3`

## What Worked Best

- ByT5 byte-level tokenization for Akkadian character noise
- Curriculum learning from noisy/broad to clean/gold data
- Joint Dropout robustness to fragmentary tablets
- Domain-specific normalization for fractions, gaps, and determinatives
- Low LR in final fine-tuning (`1e-5`) with early stopping
