import json
import itertools
from pathlib import Path

import pandas as pd
import requests
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]

SECTION_KEYS = [
    "section_1", "section_1A", "section_1B", "section_2", "section_3",
    "section_4", "section_5", "section_6", "section_7", "section_7A",
    "section_8", "section_9", "section_9A", "section_9B", "section_10",
    "section_11", "section_12", "section_13", "section_14", "section_15",
]
SECTION_TO_CODE = {key: i for i, key in enumerate(SECTION_KEYS)}
LABEL_MAP = {"positive": 0, "negative": 1}


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = ROOT_DIR / "configs" / "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def shard_url(dataset: str, revision: str, split: str, shard_id: int) -> str:
    return (
        f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/"
        f"data/large/{split}/shard_{shard_id}.jsonl"
    )


def iter_lines(response: requests.Response, chunk_size: int = 8 * 1024 * 1024):
    """Yield newline-delimited JSON lines without requests' small-chunk overhead."""
    buffer = b""
    for chunk in response.iter_content(chunk_size=chunk_size):
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if line:
                yield line
    if buffer:
        yield buffer


def stream_firms(dataset: str, revision: str, split: str, shard_ids: list[int], per_shard: int):
    """Yield up to `per_shard` firm-level JSON records from each shard."""
    for shard_id in shard_ids:
        url = shard_url(dataset, revision, split, shard_id)
        taken = 0
        print(f"Reading {split} shard {shard_id}: {url}", flush=True)
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            for raw_line in iter_lines(response):
                yield json.loads(raw_line)
                taken += 1
                if taken >= per_shard:
                    break
        print(f"  -> took {taken} firms", flush=True)
        if taken < per_shard:
            print(f"  warning: shard {shard_id} had fewer than {per_shard} firms", flush=True)


def firm_to_rows(firm: dict, split: str, counter: itertools.count) -> list[dict]:
    """Convert one firm-level JSON record into sentence-level rows."""
    rows = []
    cik = firm["cik"]

    for filing in firm.get("filings", []):
        doc_id = f"{cik}_{filing['form']}_{filing['reportDate'][:4]}"
        labels = {k: LABEL_MAP.get(v, v) for k, v in filing["labels"].items()}

        for section_key, sentences in filing.get("report", {}).items():
            if not sentences:
                continue
            section_code = SECTION_TO_CODE[section_key]
            section_name = section_key.replace("section_", "")

            for sentence_idx, sentence in enumerate(sentences):
                rows.append({
                    "cik": cik,
                    "sentence": sentence,
                    "section": section_code,
                    "section_name": section_name,
                    "labels": labels,
                    "filingDate": filing["filingDate"],
                    "name": firm.get("name"),
                    "docID": doc_id,
                    "sentenceID": f"{doc_id}_section_{section_name}_{sentence_idx}",
                    "sentenceCount": next(counter),
                    "tickers": firm.get("tickers"),
                    "exchanges": firm.get("exchanges"),
                    "entityType": firm.get("entityType"),
                    "sic": firm.get("sic"),
                    "stateOfIncorporation": firm.get("stateOfIncorporation"),
                    "tickerCount": firm.get("tickerCount"),
                    "acceptanceDateTime": filing.get("acceptanceDateTime"),
                    "form": filing.get("form"),
                    "reportDate": filing.get("reportDate"),
                    "returns": filing.get("returns"),
                    "split": split,
                })

    return rows


def download_split(
    dataset: str,
    revision: str,
    split: str,
    shard_ids: list[int],
    firms_per_shard: int,
    counter: itertools.count,
) -> pd.DataFrame:
    frames = []
    n_firms = 0
    for firm in stream_firms(dataset, revision, split, shard_ids, firms_per_shard):
        rows = firm_to_rows(firm, split, counter)
        if rows:
            frames.append(pd.DataFrame(rows))
            n_firms += 1
    print(f"{split}: {n_firms} firms -> {sum(len(f) for f in frames)} sentences")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    cfg = load_config()
    data_cfg = cfg["data"]

    out_dir = ROOT_DIR / data_cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    counter = itertools.count(1)
    dataset = data_cfg["dataset_name"]
    revision = data_cfg.get("revision", "main")

    train = download_split(
        dataset, revision, "train",
        data_cfg["train_shards"], data_cfg["firms_per_shard"]["train"], counter,
    )
    validation = download_split(
        dataset, revision, "validate",
        data_cfg["validation_shards"], data_cfg["firms_per_shard"]["validation"], counter,
    )
    test = download_split(
        dataset, revision, "test",
        data_cfg["test_shards"], data_cfg["firms_per_shard"]["test"], counter,
    )

    df = pd.concat([train, validation, test], ignore_index=True)
    output_path = out_dir / data_cfg["output_name"]
    df.to_parquet(output_path, index=False)

    print(f"\nSaved {len(df)} rows -> {output_path}")
    print(f"Firms: {df['cik'].nunique()}, filings: {df['docID'].nunique()}")
    print(df.groupby("split")["cik"].nunique())


if __name__ == "__main__":
    main()
