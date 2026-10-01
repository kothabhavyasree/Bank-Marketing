# Bank Marketing Campaign Prediction

Predicts whether a customer will subscribe to a bank term deposit (`y`).
Report: `Bank_marketing.pdf` · Exploration notebook: `notebooks/Bank_Marketing_Project_Final.ipynb`

## Layout

```
data/raw/            bank-additional-full.csv and bank-additional.csv(not tracked in git by default)
data/processed/      .npy feature matrices + dataset_metadata.json (generated)
src/                 all pipeline step scripts
pipelines/           one runner per lab (chains the scripts in src/)
models/              saved model / preprocessing artifacts (generated)
outputs/             error-analysis CSVs (generated)
artifacts/           validation reports, plots, registry report (generated)
configs/ tests/ logs/  reserved
```

## Setup

```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the labs (always from the project root)

| Lab | Command | Steps |
|-----|---------|-------|
| 3 - Baseline pipeline | `python pipelines/run_lab3_baseline.py` | `preprocess.py` -> `train.py` -> `evaluate.py` |
| 4 - MLflow tracking | `python pipelines/run_lab4_tracking.py` | `preprocess.py` -> `train_mlflow.py` -> `validate_reproducibility.py` |
| 5 - Production data pipeline | `python pipelines/run_lab5_pipeline.py` | `validate_data.py` -> `preprocess_pipeline.py` -> `validate_outputs.py` |
| 6 - Model registry | `python pipelines/run_lab6_registry.py` | `train_registry.py` -> `automate_lifecycle.py` -> `generate_registry_report.py` |

Lab 6 needs the Lab 5 outputs (`data/processed/*.npy`, `models/preprocessor.pkl`).
`train_registry.py` currently trains **V2** (200 trees, depth 15); the V1 parameters are kept
commented in the file. Run V1 first, then V2, to see the champion/challenger promotion.
Inspect runs with `mlflow ui`.

## Project-specific decisions

- Exact duplicates (12 rows) are removed before splitting.
- `duration` is dropped: it is only known after the call, so it would leak the outcome.
- `unknown` is kept as its own category.
- 80/20 stratified split, `random_state=42`; one-hot encoding + standard scaling -> 62 features.
- The target is ~89% / 11%, so the models use `class_weight="balanced"` and the registry
  promotes on **recall** rather than accuracy.
