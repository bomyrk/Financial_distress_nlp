# Financial Distress NLP

Predicting negative market reactions to SEC Form 10-K filings using classical
TF-IDF NLP. The project tests whether soft textual signals in annual reports
(MD&A, Risk Factors, Business, Financial Statements) carry predictive signal
for near-term negative market reactions at 1-day, 5-day, and 30-day horizons.

> Labels are **market-reaction proxies**, not literal accounting-distress
> labels: `0 = positive reaction`, `1 = negative reaction`.

## Project structure

```
configs/
  config.yaml                     # sampling config for download_sample.py

scripts/
  download_sample.py              # streams firm-level JSONL shards from HF -> sentence-level parquet
  build_filing_dataset.py         # aggregates sentences -> filing/section-level parquet
  config.py                       # shared config loader

src/
  modeling/
    preprocess.py                 # text cleaning, label extraction, filing/firm splits

notebooks/
  01_eda.ipynb                    # exploratory data analysis
  02_baseline.ipynb               # TF-IDF + logistic regression baseline
  03_section_horizon_experiments.ipynb  # model / section / horizon experiments
  NLP_Project_27_pipeline.ipynb   # self-contained Colab pipeline (scalable run)

data/
  samples/financial_reports_10000.parquet   # sentence-level sample
  processed/eda_ready.parquet               # preprocessed EDA table
  interim/                                    # experiment result CSVs
    classical_model_results.csv
    section_results_30d.csv
    horizon_results.csv
    horizon_results_base_logreg.csv
    label_prevalence.csv

models/
  logreg_baseline.pkl             # trained baseline classifier
  tfidf_vectorizer.pkl            # fitted TF-IDF vectorizer

reports/
  financial_distress_nlp_report.md   # full write-up with all results
  Financial Distress Prediction.pdf / .docx
  NLP Project.pptx                   # presentation deck
  figures/                           # ROC/PR curves and confusion matrices
```

## Data

Source: [`JanosAudran/financial-reports-sec`](https://huggingface.co/datasets/JanosAudran/financial-reports-sec)
on Hugging Face (~4,677 firms, ~55k filings, ~72M sentences). The full corpus
is too large for local runs, so `scripts/download_sample.py` streams selected
JSONL shards and converts nested filing/section/sentence records into
sentence-level rows. Shards and firms-per-shard are controlled by
`configs/config.yaml`, producing a firm-diverse sample with **zero firm
overlap** across the official `train` / `validate` / `test` splits (splits are
by `cik`, preventing company-level leakage).

## Pipeline

### 1. Build the sentence-level sample

```bash
python scripts/download_sample.py
# -> data/samples/financial_reports_firm_sample.parquet
```

### 2. Build the filing-level dataset

```bash
python scripts/build_filing_dataset.py
# -> data/processed/filing_level.parquet
```

Streams the sentence-level parquet in 100k-row batches, cleans text, extracts
`label_1d` / `label_5d` / `label_30d` (and returns when available), and writes
one row per filing × section for sections `ALL`, `1` (Business), `1A` (Risk
Factors), `2` (Properties), `7` (MD&A), `8` (Financial Statements). Filings
with fewer than 10 sentences are dropped.

### 3. Experiments

Run the notebooks in order:

| notebook | purpose |
|---|---|
| `01_eda.ipynb` | label prevalence, section/horizon distributions |
| `02_baseline.ipynb` | TF-IDF + balanced logistic regression baseline |
| `03_section_horizon_experiments.ipynb` | classical model comparison, section ablation, horizon sweep (PR-AUC + ROC-AUC) |
| `NLP_Project_27_pipeline.ipynb` | self-contained Google Colab pipeline: streams shards, builds filing-level rows incrementally, runs the same experiments, and exports CSVs/PNGs/report artifacts |

### Models

All classical TF-IDF pipelines (CPU-safe, no GPU required):

- `base_logreg` — `TfidfVectorizer(max_features=10000, ngram_range=(1,2))` + `LogisticRegression(class_weight="balanced")`
- `improved_logreg` — larger vocabulary, `min_df=3`, `max_df=0.9`, `sublinear_tf=True`
- `unbalanced_logreg` — improved TF-IDF without class weighting
- `complement_nb` — improved TF-IDF (`norm=None`) + `ComplementNB`
- `linearsvc` — improved TF-IDF + balanced `LinearSVC` (decision scores used for AUC metrics)

Primary metric: **PR-AUC** (labels are near-balanced but not perfectly so);
ROC-AUC, F1, precision, recall, and accuracy are also reported.

## Results summary

Full details and tables: [`reports/financial_distress_nlp_report.md`](reports/financial_distress_nlp_report.md)

Key findings from the local pilot (84 firms, 1,131 filings):

- All models sit modestly above the majority baseline (PR-AUC ≈ label-1 prevalence ≈ 0.53).
- **Item 7 / MD&A is the strongest individual section** on test (PR-AUC 0.655).
- **5-day horizon is easiest to rank** for ALL sections (test PR-AUC 0.684);
  30-day is more meaningful and better balanced.
- Validation/test rankings disagree on small eval sets — treat results as
  exploratory, not confirmatory.

## Reproducibility

```bash
python scripts/download_sample.py        # optional: rebuild sample
python scripts/build_filing_dataset.py   # build filing-level parquet
# then run notebooks/01–03
```

For the scaled run, upload `notebooks/NLP_Project_27_pipeline.ipynb` to
Google Colab and run all cells; outputs are written to
`/content/financial_distress_outputs/`.

## Limitations

- Labels proxy market reactions, which can be driven by market-wide factors
  rather than firm-specific distress.
- Sample is a subset of the full corpus; evaluation sets are small, so
  section/horizon rankings are noisy.
- Classical models only — transformer fine-tuning (e.g. FinBERT) is out of
  scope for this project.
