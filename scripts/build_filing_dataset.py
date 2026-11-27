import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.modeling.preprocess import preprocess

RAW_PATH = ROOT_DIR / "data" / "samples" / "financial_reports_firm_sample.parquet"
OUT_PATH = ROOT_DIR / "data" / "processed" / "filing_level.parquet"
KEY_SECTIONS = ["1A", "7", "8", "1", "2"]
BATCH_SIZE = 100_000

META_COLS = [
    "cik", "name", "filingDate", "split",
    "label_1d", "label_5d", "label_30d",
    "ret_1d", "ret_5d", "ret_30d",
]


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    parts = defaultdict(list)   # (docID, section_name) -> cleaned sentences
    meta = {}                   # (docID, section_name) -> first-row metadata
    print(f"Streaming {RAW_PATH} in batches of {BATCH_SIZE:,}")

    pf = pq.ParquetFile(RAW_PATH)
    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE)):
        df = preprocess(batch.to_pandas())
        for (doc_id, section_name), grp in df.groupby(["docID", "section_name"], sort=False):
            key = (doc_id, section_name)
            if key not in meta:
                meta[key] = grp.iloc[0][META_COLS].to_dict()
            parts[key].extend(grp["sentence"].tolist())
        print(f"  batch {batch_idx}: cumulative keys={len(parts):,}", flush=True)
        del df

    print(f"Building filing-section rows from {len(parts):,} groups...")
    section_rows = []
    for (doc_id, section_name), sentences in parts.items():
        row = {"docID": doc_id, "section_name": section_name}
        row.update(meta[(doc_id, section_name)])
        row["text"] = " ".join(sentences)
        row["n_sentences"] = len(sentences)
        section_rows.append(row)

    section_df = pd.DataFrame(section_rows)

    print("Building ALL-sections rows...")
    all_df = (
        section_df.groupby("docID", sort=False)
        .agg(
            text=("text", lambda x: " ".join(x)),
            n_sentences=("n_sentences", "sum"),
            cik=("cik", "first"),
            name=("name", "first"),
            filingDate=("filingDate", "first"),
            split=("split", "first"),
            label_1d=("label_1d", "first"),
            label_5d=("label_5d", "first"),
            label_30d=("label_30d", "first"),
            ret_1d=("ret_1d", "first"),
            ret_5d=("ret_5d", "first"),
            ret_30d=("ret_30d", "first"),
        )
        .reset_index()
    )
    all_df["section_name"] = "ALL"

    keep_sections = set(KEY_SECTIONS) | {"ALL"}
    out = pd.concat(
        [all_df, section_df[section_df["section_name"].isin(keep_sections)]],
        ignore_index=True,
    )
    out.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved {len(out):,} filing-section rows -> {OUT_PATH}")
    print(out.groupby(["section_name", "split"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
