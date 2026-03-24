# Akkadian–English (Deep Past) — scripts

- `trainv4.py` — entrenamiento ByT5 (fases CDLI, OARE, gold, etc.; datos en Drive/Kaggle).
- `infere.py` — inferencia / submission.
- `gemini_pseudo_labeler.py` — pseudo-etiquetas con Gemini.
- `gemini_back_translator.py` — back-translation sintética.
- `Joint_Dropout-main/` — pipeline Joint Dropout (sin `venv2`; instalar deps con el README interno).

Datos grandes (`publications.csv`, etc.) no van en el repo: usar competencia Kaggle u ORACC.

## Subir a GitHub

Desde esta carpeta (`facil/`):

1. Crea un repositorio **vacío** en GitHub (sin README ni .gitignore), por ejemplo `akkadian-dpc-tools`.
2. Conecta y sube:

```bash
cd /ruta/a/balosento/facil
git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
git push -u origin main
```

Si GitHub te pide autenticación, usa un **Personal Access Token** (Settings → Developer settings) como contraseña, o `gh auth login`.

Ajusta autor del commit si hace falta:

```bash
git config user.name "Tu Nombre"
git config user.email "tu@email o id+username@users.noreply.github.com"
git commit --amend --reset-author --no-edit
```
