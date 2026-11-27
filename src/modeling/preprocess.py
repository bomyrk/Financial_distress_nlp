import re
import pandas as pd
import numpy as np

def clean_text(text: str) -> str:
    """Remove HTML artifacts, normalize whitespace."""
    text = re.sub(r"<[^>]+>", "", text)       # HTML tags
    text = re.sub(r"\\n|\\t|\\r", " ", text)  # escape chars
      # --- Remove dates ---
    # "May 31, 2020" / "31 May 2020" / "2020-05-31"
    text = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b', ' ', text)
    text = re.sub(r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4}\b', ' ', text)
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', ' ', text)
    # "fiscal year 2020" / "fiscal 2019" / "year 2020"
    text = re.sub(r'\b(?:fiscal\s+)?(?:year|20\d{2}|19\d{2})\s+(?:20\d{2}|19\d{2})\b', ' ', text)
    text = re.sub(r'\b(?:FISCAL\s+)?(?:year|20\d{2}|19\d{2})\s+(?:20\d{2}|19\d{2})\b', ' ', text)
    # Standalone years (1900–2099)
    text = re.sub(r'\b(?:19|20)\d{2}\b', ' ', text)

    text = re.sub(r'[^a-zA-Z0-9\s\.\$%\-\(\)]', ' ', text)
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    text = text.lower()
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    df = df.copy()

    # Clean text
    df["sentence"] = df["sentence"].apply(clean_text)

    # Drop empty / too-short sentences
    df = df[df["sentence"].str.len() > 10].reset_index(drop=True)

    # Extract labels (0=positive, 1=negative)
    for horizon in ["1d", "5d", "30d"]:
        df[f"label_{horizon}"] = df["labels"].apply(lambda x: x[horizon])

    # Extract returns when available
    if "returns" in df.columns:
        for horizon in ["1d", "5d", "30d"]:
            df[f"ret_{horizon}"] = df["returns"].apply(lambda x: x[horizon]["ret"])

    # Section name: prefer the explicit column produced by download_sample.py.
    if "section_name" not in df.columns:
        # Order matches the small_full/_REPORT_KEYS section ordering.
        section_map = {
            0: "1", 1: "1A", 2: "1B", 3: "2", 4: "3", 5: "4",
            6: "5", 7: "6", 8: "7", 9: "7A", 10: "8", 11: "9",
            12: "9A", 13: "9B", 14: "10", 15: "11", 16: "12",
            17: "13", 18: "14", 19: "15",
        }
        df["section_name"] = df["section"].map(section_map)

    # Word count
    df["word_count"] = df["sentence"].str.split().str.len()

    return df

def aggregate_by_filing(df: pd.DataFrame, section: str = None) -> pd.DataFrame:
    """Aggregate sentences into one document per filing."""
    if section is not None:
        df = df[df["section_name"] == section]

    agg_spec = {
        "text": ("sentence", lambda x: " ".join(x)),
        "label_30d": ("label_30d", "first"),
        "n_sentences": ("sentence", "count"),
    }
    for col in [
        "cik", "name", "filingDate", "split",
        "label_1d", "label_5d", "ret_1d", "ret_5d", "ret_30d",
    ]:
        if col in df.columns:
            agg_spec[col] = (col, "first")

    agg = df.groupby("docID").agg(**agg_spec).reset_index()

    # Drop filings with too few sentences
    agg = agg[agg["n_sentences"] >= 10].reset_index(drop=True)
    return agg


def doc_id_split(df: pd.DataFrame, test_ratio: float = 0.2,
                 random_state: int = 42):
    """Split by docID to prevent sentence-level leakage."""
    unique_docs = df["docID"].unique()
    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_docs)

    n_test = max(1, int(len(unique_docs) * test_ratio))
    test_docs = set(unique_docs[:n_test])
    train_docs = set(unique_docs[n_test:])

    train = df[df["docID"].isin(train_docs)].reset_index(drop=True)
    test  = df[df["docID"].isin(test_docs)].reset_index(drop=True)

    print(f"Train: {len(train)} rows, {len(train_docs)} filings")
    print(f"Test:  {len(test)} rows, {len(test_docs)} filings")
    return train, test


def firm_level_split(df: pd.DataFrame, val_size: float = 0.15,
                     test_size: float = 0.2, random_state: int = 42):
    """Split filing-level data by company (cik) to avoid firm leakage."""
    if "cik" not in df.columns:
        raise ValueError("firm_level_split requires a 'cik' column")

    firm_stats = (
        df.groupby("cik")
        .agg(label_mean=("label_30d", "mean"), n_filings=("docID", "nunique"))
        .reset_index()
    )
    firm_stats["stratum"] = (firm_stats["label_mean"] >= 0.5).astype(int)

    rng = np.random.default_rng(random_state)
    firm_to_split = {}
    for stratum, group in firm_stats.groupby("stratum"):
        firms = group["cik"].to_numpy().copy()
        rng.shuffle(firms)
        n = len(firms)
        n_test = max(1, int(round(n * test_size))) if n >= 5 else 0
        n_val = max(1, int(round(n * val_size))) if n >= 10 else 0
        n_val = min(n_val, max(0, n - n_test - 1))
        n_test = min(n_test, max(0, n - n_val - 1))

        test_firms = firms[:n_test]
        val_firms = firms[n_test:n_test + n_val]
        train_firms = firms[n_test + n_val:]
        for cik in train_firms:
            firm_to_split[cik] = "train"
        for cik in val_firms:
            firm_to_split[cik] = "validation"
        for cik in test_firms:
            firm_to_split[cik] = "test"

    df = df.copy()
    df["split"] = df["cik"].map(firm_to_split)

    train = df[df["split"] == "train"].drop(columns="split").reset_index(drop=True)
    val = df[df["split"] == "validation"].drop(columns="split").reset_index(drop=True)
    test = df[df["split"] == "test"].drop(columns="split").reset_index(drop=True)

    print(f"Train: {len(train)} filings, {train['cik'].nunique()} firms")
    print(f"Val:   {len(val)} filings, {val['cik'].nunique()} firms")
    print(f"Test:  {len(test)} filings, {test['cik'].nunique()} firms")
    return train, val, test


def train_val_test_split(df: pd.DataFrame, val_size: float = 0.15,
                         test_size: float = 0.2, random_state: int = 42):
    """Use official split column when present, otherwise split by firm."""
    if "split" in df.columns:
        train = df[df["split"] == "train"].reset_index(drop=True)
        val = df[df["split"].isin(["validation", "val", "validate"])].reset_index(drop=True)
        test = df[df["split"] == "test"].reset_index(drop=True)
        print(f"Train: {len(train)} filings, {train['cik'].nunique()} firms")
        print(f"Val:   {len(val)} filings, {val['cik'].nunique()} firms")
        print(f"Test:  {len(test)} filings, {test['cik'].nunique()} firms")
        return train, val, test
    return firm_level_split(df, val_size=val_size, test_size=test_size,
                            random_state=random_state)
