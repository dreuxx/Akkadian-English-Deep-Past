# Akkadian–English (Deep Past) — scripts

- `trainv4.py` — entrenamiento ByT5 (fases CDLI, OARE, gold, etc.; datos en Drive/Kaggle).
- `infere.py` — inferencia / submission.
- `gemini_pseudo_labeler.py` — pseudo-etiquetas con Gemini.
- `gemini_back_translator.py` — back-translation sintética.
- `Joint_Dropout-main/` — pipeline Joint Dropout (sin `venv2`; instalar deps con el README interno).

Datos grandes (`publications.csv`, etc.) no van en el repo: usar competencia Kaggle u ORACC.
