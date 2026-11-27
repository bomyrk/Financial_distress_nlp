# Financial Distress Prediction from SEC 10-K Soft Data

**Pilot + scaled Colab run — firm-diverse samples, classical NLP baseline**

## 1. Objective

This project tests whether soft textual signals in SEC Form 10-K filings can predict near-term negative market reactions. The practical goal is a defensible NLP pipeline that:

- aggregates sentence-level filing text into filing-level documents,
- avoids company leakage across train/validation/test splits,
- compares predictive signal across 10-K sections and prediction horizons,
- reports ranking metrics suitable for mildly imbalanced binary labels.

The labels are **market-reaction proxies**, not literal accounting-distress labels. A label of `1` means the negative class for the market-reaction target defined by the dataset; throughout the metrics below, class `1` is treated as the positive class for precision/recall/F1/PR-AUC.

## 2. Data

This section describes the local pilot sample used for sections 4.1–4.4; the scaled Colab sample is described in section 4A.1.

### 2.1 Source dataset

We use the Hugging Face dataset `JanosAudran/financial-reports-sec`, which contains SEC annual reports grouped into filings and sentences. The dataset card reports roughly 4,677 firms, 55,349 filings, and ~72M sentences. Because the full corpus is too large for this machine and deadline, we evaluate on a **firm-diverse sample** rather than the first-N-rows sample.

### 2.2 Sampling strategy

`scripts/download_sample.py` streams firm-level JSONL shards and converts nested filing/section/sentence structures into sentence-level rows. The sample is controlled by `configs/config.yaml`:

| split | shards | firms/shard | firms | role |
|---|---:|---:|---:|---|
| train | 0, 3, 6, 9 | 12 | 48 | model fitting |
| validate | 0, 5 | 8 | 16 | model/section selection |
| test | 1, 7 | 10 | 20 | final reporting only |

This produced `data/samples/financial_reports_firm_sample.parquet` with:

- **1,343,102 sentence rows**
- **84 firms**
- **1,131 filings**
- **zero firm overlap** across train/validation/test

### 2.3 Filing-level dataset

`scripts/build_filing_dataset.py` streams the sentence-level parquet in batches of 100,000 rows, applies `src/modeling/preprocess.py`, and writes `data/processed/filing_level.parquet`.

After filtering to filing-section rows with at least 10 sentences, the reusable table has **5,047 rows** across sections `ALL`, `1`, `1A`, `2`, `7`, and `8`.

| section | meaning | train filings | validation filings | test filings |
|---|---|---:|---:|---:|
| ALL | all 10-K sections combined | 977 | 42 | 112 |
| 1 | Business | 887 | 39 | 109 |
| 1A | Risk Factors | 623 | 40 | 92 |
| 2 | Properties | 406 | 10 | 26 |
| 7 | MD&A | 792 | 41 | 111 |
| 8 | Financial Statements | 626 | 23 | 91 |

The `ALL` split counts are train = 977 filings / 48 firms, validation = 42 filings / 16 firms, test = 112 filings / 20 firms.

### 2.4 Labels and horizons

The dataset provides market-reaction labels for three horizons:

- `label_1d`: short-horizon reaction
- `label_5d`: medium-horizon reaction
- `label_30d`: longer-horizon reaction

Label convention: `0 = positive`, `1 = negative`. For the `ALL` filings, the share of label `1` is close to balanced but varies by split and horizon:

| horizon | train label-1 rate | validation label-1 rate | test label-1 rate |
|---|---:|---:|---:|
| 1d | 0.526 | 0.524 | 0.580 |
| 5d | 0.511 | 0.595 | 0.598 |
| 30d | 0.459 | 0.571 | 0.527 |

Because the classes are not perfectly balanced and PR-AUC is more informative than accuracy under imbalance, we report **PR-AUC** as the primary ranking metric alongside ROC-AUC, F1, precision, recall, and accuracy.

## 3. Methodology

### 3.1 Text preprocessing

`clean_text` in `src/modeling/preprocess.py`:

- removes HTML tags and escaped newline/tab characters,
- removes explicit dates and standalone years,
- keeps alphanumeric characters plus a small set of financial symbols,
- lowercases text and collapses whitespace,
- drops sentences shorter than 10 characters after cleaning.

### 3.2 Aggregation

Sentences are aggregated by `docID` and `section_name`. For each filing-section pair, sentences are concatenated into one document. `ALL` is built by concatenating all available sections for the same filing. Filings/sections with fewer than 10 sentences are excluded.

### 3.3 Splitting and leakage control

We use the dataset’s official firm-level `train` / `validate` / `test` split column. Because the split is by firm (`cik`), the same company cannot appear in both training and evaluation. This avoids the main leakage mode in financial NLP: learning company-specific style or boilerplate rather than generalizable distress language.

### 3.4 Models

All models are classical TF-IDF pipelines, chosen because the environment has no usable CUDA GPU and filing documents are long.

Base configuration:

- `TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words="english", min_df=2)`
- `LogisticRegression(class_weight="balanced", max_iter=1000)`

Compared variants in `notebooks/03_section_horizon_experiments.ipynb`:

| model | description |
|---|---|
| `base_logreg` | base TF-IDF + balanced logistic regression |
| `improved_logreg` | larger TF-IDF vocabulary, `min_df=3`, `max_df=0.9`, `sublinear_tf=True` |
| `unbalanced_logreg` | improved TF-IDF without class weighting |
| `complement_nb` | improved TF-IDF with `norm=None` + `ComplementNB` |
| `linearsvc` | improved TF-IDF + balanced `LinearSVC` |

For `LinearSVC`, decision scores are used for ROC-AUC/PR-AUC because the model has no `predict_proba`.

### 3.5 Evaluation protocol

1. Model selection uses **validation PR-AUC** on `ALL` / `label_30d`.
2. Section selection uses validation PR-AUC with a minimum of 20 validation filings to avoid unstable tiny-section comparisons.
3. Test metrics are reported once as final evidence, but because many section/horizon combinations were inspected, the test results should be interpreted as exploratory rather than a single pre-registered confirmatory test.
4. A majority baseline has PR-AUC approximately equal to the label-1 prevalence; ROC-AUC baseline is 0.5.

## 4. Local pilot results (`03` + `data/interim`)

### 4.1 Classical model comparison

Setting: `ALL` sections, `label_30d`.

| model | validation PR-AUC | test PR-AUC | test ROC-AUC | test F1 | test accuracy |
|---|---:|---:|---:|---:|---:|
| base_logreg | **0.615** | 0.617 | 0.586 | 0.602 | 0.563 |
| improved_logreg | 0.596 | 0.631 | 0.588 | 0.597 | 0.554 |
| unbalanced_logreg | 0.609 | 0.630 | 0.589 | 0.477 | 0.589 |
| complement_nb | 0.571 | 0.561 | 0.547 | 0.626 | 0.563 |
| linearsvc | 0.527 | **0.643** | 0.581 | 0.585 | 0.545 |

Interpretation:

- Validation selected `base_logreg`.
- On test, `linearsvc` has the highest PR-AUC, but the validation/test disagreement is expected with only 42 validation filings.
- All classical models are only modestly above the majority baseline: for `ALL`/`30d`, the test label-1 prevalence is 0.527, so a constant predictor has PR-AUC ≈ 0.527.

### 4.2 Section ablation

Setting: `label_30d`, model = validation-selected `base_logreg`.

| section | validation PR-AUC | test PR-AUC | test ROC-AUC | test F1 | test accuracy | note |
|---|---:|---:|---:|---:|---:|---|
| Item 7 — MD&A | 0.542 | **0.655** | **0.640** | **0.631** | **0.631** | best overall on test |
| ALL | **0.615** | 0.617 | 0.586 | 0.602 | 0.563 | best validation PR-AUC |
| Item 8 — Financial Statements | 0.412 | 0.584 | 0.520 | 0.463 | 0.516 | weak |
| Item 1 — Business | 0.537 | 0.583 | 0.582 | 0.495 | 0.550 | weak |
| Item 1A — Risk Factors | 0.550 | 0.582 | 0.577 | 0.533 | 0.543 | weaker than expected |
| Item 2 — Properties | 0.730 | 0.536 | 0.494 | 0.480 | 0.500 | only 10 val / 26 test filings — unreliable |

Key finding: **Item 7 / MD&A is the strongest individual section on test**, while `ALL` is the most stable choice by validation. Item 1A does **not** beat MD&A on this sample. Item 2 should not drive conclusions because its validation and test sets are tiny.

### 4.3 Horizon comparison

The `03` horizon sweep uses the validation-selected sections (`ALL`, `1A`, `7`, `1`) and `base_logreg`, written to `data/interim/horizon_results.csv`.

#### ALL sections

| horizon | validation PR-AUC | test PR-AUC | test ROC-AUC | test F1 | test accuracy |
|---|---:|---:|---:|---:|---:|
| 1d | 0.618 | 0.647 | 0.503 | 0.444 | 0.509 |
| 5d | 0.598 | **0.684** | 0.568 | 0.574 | 0.536 |
| 30d | **0.615** | 0.617 | **0.586** | **0.602** | **0.563** |

#### Item 7 — MD&A

| horizon | validation PR-AUC | test PR-AUC | test ROC-AUC | test F1 | test accuracy |
|---|---:|---:|---:|---:|---:|
| 1d | 0.680 | 0.652 | 0.523 | 0.509 | 0.532 |
| 5d | 0.645 | 0.639 | 0.558 | 0.532 | 0.541 |
| 30d | 0.542 | **0.655** | **0.640** | **0.631** | **0.631** |

#### All horizon test rows (PR-AUC)

| section | horizon | test PR-AUC | test ROC-AUC | test F1 | test accuracy |
|---|---|---:|---:|---:|---:|
| ALL | 5d | **0.684** | 0.568 | 0.574 | 0.536 |
| Item 7 | 30d | 0.656 | 0.640 | 0.631 | 0.631 |
| Item 7 | 1d | 0.652 | 0.523 | 0.509 | 0.532 |
| ALL | 1d | 0.647 | 0.503 | 0.444 | 0.509 |
| Item 7 | 5d | 0.639 | 0.558 | 0.532 | 0.541 |
| Item 1 | 5d | 0.635 | 0.544 | 0.400 | 0.477 |
| ALL | 30d | 0.617 | 0.586 | 0.602 | 0.563 |
| Item 1 | 1d | 0.593 | 0.497 | 0.396 | 0.495 |
| Item 1A | 5d | 0.588 | 0.516 | 0.646 | 0.511 |
| Item 1 | 30d | 0.583 | 0.582 | 0.495 | 0.550 |
| Item 1A | 30d | 0.581 | 0.577 | 0.533 | 0.543 |
| Item 1A | 1d | 0.544 | 0.479 | 0.536 | 0.511 |

Interpretation:

- **5d is the easiest horizon to rank** for `ALL` (test PR-AUC 0.684).
- **Item 7 / MD&A remains strong** at 30d and 1d (test PR-AUC 0.656 and 0.652), supporting the local-pilot finding that MD&A carries useful soft signals.
- **Item 1A is weaker across horizons** on this pilot (best test PR-AUC 0.588 at 5d).
- **Item 1 is mid-pack**; its best horizon is 5d (0.635) but F1/accuracy are low.
- The best overall balance is still **Item 7 + 30d** locally, but section 4A shows this does not clearly replicate at larger scale.

### 4.4 Majority-baseline comparison

For a constant predictor, PR-AUC ≈ label-1 prevalence and ROC-AUC = 0.5. Selected comparisons:

| setting | test label-1 prevalence | model PR-AUC | lift over majority |
|---|---:|---:|---:|
| ALL / 30d / base_logreg | 0.527 | 0.617 | +0.090 |
| ALL / 5d / base_logreg | 0.598 | 0.684 | +0.086 |
| Item 7 / 30d / base_logreg | 0.522 | 0.656 | +0.133 |
| Item 7 / 1d / base_logreg | 0.577 | 0.652 | +0.075 |
| Item 1 / 5d / base_logreg | 0.596 | 0.635 | +0.039 |

The lifts are real but modest. They support the claim that filing text contains signal, not that the model is production-ready.

## 4A. Scaled Colab run (`05` / `financial_name (2)` outputs; primary evidence)

A larger CPU run was executed in Google Colab from `notebooks/04_colab_full_pipeline.ipynb`; the executed copy with outputs is `notebooks/05_full_pipeline.ipynb`. It streamed raw firm-level JSONL shards and built filing-level rows incrementally. Item 2 was removed from the standalone section ablation; it remains inside `ALL`.

### 4A.1 Sample

| split | firms | ALL filings | filing-section rows after >=10 sentences |
|---|---:|---:|---:|
| train | 180 | 3,813 | 14,764 |
| validate | 80 | 233 | 1,029 |
| test | 120 | 575 | 2,583 |

Requested shards were 0–11, but train/test shards 10–11 were unavailable, so train/test used 10 shards each; validation used shards 0–7. Section counts after filtering (train/test): ALL 3,813/575; Item 1 3,506/559; Item 7 2,972/574; Item 1A 2,299/498; Item 8 2,174/377. Validation section counts: ALL 233, Item 1A 206, Item 1 226, Item 7 232, Item 8 132.

### 4A.2 Model comparison — ALL / 30d

Validation selected `linearsvc` (PR-AUC 0.692), but test PR-AUC was highest for `improved_logreg` (0.628), followed closely by `unbalanced_logreg` (0.627), `linearsvc` (0.618), `base_logreg` (0.598), and `complement_nb` (0.557). The validation/test gap persists; PR-AUC differences among the top three are small for 575 test filings.

### 4A.3 Section ablation — 30d, LinearSVC

Validation leaders: Item 1A 0.703 (n=206), ALL 0.692 (n=233), Item 1 0.642 (n=226), Item 8 0.633 (n=132), Item 7 0.607 (n=232). Test PR-AUC: ALL 0.618, Item 8 0.584, Item 1 0.571, Item 7 0.558, Item 1A 0.542. Thus validation chose Item 1A, but test favored ALL; section ranking remains unstable. Item 7 is again mid-pack, so the earlier local pilot finding is not confirmed at larger scale.

### 4A.4 Horizon sweep — ALL / Item 1A / Item 1, LinearSVC

The scaled horizon sweep covered the validation-selected sections (`ALL`, `1A`, `1`). Test PR-AUC: ALL/1d 0.669, ALL/5d 0.662, Item 1/1d 0.649, ALL/30d 0.618, Item 1A/1d 0.584, Item 1A/5d 0.575, Item 1/30d 0.571, Item 1/5d 0.559, Item 1A/30d 0.542. The shortest horizons are easiest to rank; 1d is slightly ahead of 5d, while 30d is lower but still above the majority baseline.

### 4A.5 Figures and artifacts

The run saved ROC/PR curves for ALL/30d, ALL/5d, and Item 1A/30d, plus a confusion matrix for ALL/30d, and wrote CSV summaries under `/content/financial_distress_outputs/`. PPTX/download cells are ignored for the report conclusions; use the CSV/PNG experiment outputs as evidence.

## 5. Discussion

### What the results support

- Filing-level aggregation is necessary; sentence-level labels would leak document context.
- Firm-level splits are essential; random filing/sentence splits would overstate performance.
- On the larger Colab sample (380 firms, 3,813 ALL filings), classical TF-IDF still extracts modest signal; ALL/1d and ALL/5d are easiest to rank by PR-AUC, while 30d is lower but still above the majority baseline.
- Section evidence is unstable: validation chose Item 1A, but test favored ALL; Item 7 is not confirmed at larger scale, and Item 2 was removed from the ablation to avoid small-section artifacts.

### What the results do not support

- They do **not** establish state-of-the-art performance.
- They do **not** prove literal financial-distress prediction; the target is market reaction.
- They do **not** justify fine-tuning FinBERT; the Colab run remained CPU-safe and classical.
- They do **not** establish tight generalization bounds; section subsets remain smaller than ALL.

## 6. Limitations

1. **Sample size**: the scaled Colab run uses 380 firms and 3,813 ALL filings, much larger than the 84-firm local pilot but still a subset of the full dataset; requested train/test shards 10–11 were unavailable, so those splits used 10 shards each.
2. **Small evaluation sets**: Colab `ALL` validation has 233 filings and test has 575 filings; section subsets are smaller (e.g., Item 8 validation = 132, test = 377).
3. **Exploratory test usage**: many section/horizon/model combinations were inspected, so test metrics are optimistic relative to a single pre-registered experiment.
4. **Label proxy**: labels reflect market returns around filing dates, which can be driven by market-wide moves, news, or microstructure rather than firm-specific distress.
5. **Classical models only**: transformer fine-tuning was not pursued within the course scope.
6. **Horizon coverage**: the scaled horizon sweep covered only ALL/Item 1A/Item 1; the local pilot added Item 7 and Item 1, but Item 8 horizons are only represented by the 30d section ablation.

## 7. Recommended next steps

If time permits, prioritize analyses that strengthen the report rather than new model families:

1. Add a formal majority/positive-rate baseline table to the notebook outputs.
2. Expand the firm sample with more shards if memory/time allow.
3. Consider chunk-level TF-IDF or sentence-averaged probabilities before any transformer experiment.
4. Keep transformer fine-tuning out of scope unless explicitly requested; BERT/FinBERT is planned for the next NLP course.

## 8. Reproducibility

Environment: `secmlf25` conda environment, CPU-only.

Commands:

```bash
cd "/home/kamguia/Documents/AIM2/Info 6148 Natural Language Processing/Financial_distress_nlp"
conda activate secmlf25

# optional: rebuild sentence-level sample
python scripts/download_sample.py

# build filing-level dataset
python scripts/build_filing_dataset.py
```

Colab:

- Upload `notebooks/04_colab_full_pipeline.ipynb` to Google Colab and run all cells.
- The executed notebook with outputs is `notebooks/05_full_pipeline.ipynb`.
- Colab outputs are written to `/content/financial_distress_outputs/` and downloaded as `financial_distress_report.pptx`, `report_summary.md`, CSVs, and PNG figures.
- The CSV/PNG outputs used for section 4A are in `notebooks/financial_name (2)/financial_distress_outputs/`.

Notebooks:

- `notebooks/05_full_pipeline.ipynb` — executed Colab run with outputs (primary evidence)
- `notebooks/04_colab_full_pipeline.ipynb` — standalone Colab pipeline
- `notebooks/03_section_horizon_experiments.ipynb` — local pilot experiments
- `notebooks/02_baseline.ipynb` — optional baseline smoke test/artifacts
- `notebooks/01_eda.ipynb` — optional EDA

Outputs:

Local pilot:

- `data/processed/filing_level.parquet`
- `data/interim/classical_model_results.csv`
- `data/interim/section_results_30d.csv`
- `data/interim/horizon_results.csv`
- `data/interim/label_prevalence.csv`

Colab run (`/content/financial_distress_outputs/`):

- `financial_distress_report.pptx`
- `report_summary.md`
- `classical_model_results.csv`
- `section_results_30d.csv`
- `horizon_results.csv`
- `label_prevalence.csv`
- `figures/roc_pr_ALL_30d.png`
- `figures/roc_pr_ALL_5d.png`
- `figures/roc_pr_1A_30d.png`
- `figures/confusion_ALL_30d.png`

## 9. Conclusion

On the larger firm-diverse Colab sample with leakage-safe splits, classical TF-IDF models extract modest but real signal from SEC 10-K text. The clearest ranking results are **ALL sections / 1-day** (PR-AUC 0.669) and **ALL sections / 5-day** (PR-AUC 0.662), while 30-day is lower (PR-AUC 0.618) but still above the majority baseline. Section evidence is unstable: validation chose Item 1A, but test favored ALL, and Item 7 is not confirmed at larger scale. Present the findings as exploratory evidence that filing text contains signal, not as a production-ready distress classifier.

## Appendix A. Result files

| file | contents |
|---|---|
| `notebooks/05_full_pipeline.ipynb` | executed Colab run with outputs |
| `notebooks/financial_name (2)/financial_distress_outputs/` | latest Colab CSV/PNG outputs used in section 4A |
| `financial_distress_report.pptx` | Colab-generated submission deck |
| `report_summary.md` | Colab-generated markdown summary |
| `classical_model_results.csv` | model comparison on ALL/30d |
| `section_results_30d.csv` | section ablation on 30d |
| `horizon_results.csv` | horizon sweep CSV (local pilot: ALL/1A/7/1; Colab: ALL/1A/1) |
| `label_prevalence.csv` | label-1 prevalence by section/horizon/split |
| `roc_pr_ALL_30d.png`, `roc_pr_ALL_5d.png`, `roc_pr_1A_30d.png` | Colab ROC/PR curves |
| `confusion_ALL_30d.png` | Colab confusion matrix |

## Appendix B. Interpretation guide for labels

- `label_* = 0`: positive market reaction
- `label_* = 1`: negative market reaction
- Metrics treat `1` as the positive class.
- `ret_1d`, `ret_5d`, `ret_30d` are preserved in the filing-level table when available.

## Appendix C. Dataset card reference

- Dataset: `JanosAudran/financial-reports-sec`
- URL: https://huggingface.co/datasets/JanosAudran/financial-reports-sec
- Reported scale: ~4,677 firms, ~55,349 filings, ~72M sentences.
- This report uses a sampled subset, not the full corpus.

---

*Draft report generated from the completed classical experiments. Update tables if notebooks are rerun after expanded sampling or model changes.*
