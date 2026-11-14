from datasets import load_dataset
from itertools import islice
import pandas as pd
import os
from config import load_config

cfg = load_config()

N = cfg["data"]["n_samples"]
OUT_DIR = cfg["data"]["out_dir"]
os.makedirs(OUT_DIR, exist_ok=True)

ds = load_dataset(
    cfg["data"]["dataset_name"],
    revision=cfg["data"]["revision"],
    streaming=True,
)

records = list(islice(ds["train"], N))
df = pd.DataFrame(records)

parquet_path = os.path.join(OUT_DIR, f"financial_reports_{N}.parquet")
df.to_parquet(parquet_path, index=False)
print(f"Saved {len(df)} rows → {parquet_path}")
print(df.head())   
