"""Build cross-scale supplementary experiment data.

Downloads the same three public datasets used in the paper (Alpaca, GSM8K,
CodeXGLUE-Python), applies the paper's document formatters, and emits one
JSONL file per domain with deterministic source shards.

Configuration (supplementary setting):
    train docs per domain : 120   (repository default; the paper's Section 4
                                    smoke run used 12 -- we enlarge it for
                                    rank-statistics power, 6 shards instead
                                    of 4, and document this deviation)
    val   docs per domain : 32    (paper used 6; enlarged for stable NLL)
    source shards         : 6     (paper used 4)

Output: supplementary/data/<domain>.jsonl with fields
    doc_id, source_id, text, label, domain, split
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_pricing.real_datasets import REAL_DATASET_SPECS  # noqa: E402

DOMAIN_KEYS = ["alpaca_instruction", "gsm8k_math", "codexglue_python"]

TRAIN_LIMIT = 120
VAL_LIMIT = 32
SHARDS = 6
SEED = 17

# Column aliases so that raw rows from any export of CodeSearchNet/CodeXGLUE
# map onto the keys expected by the paper's formatter.
CODE_CODE_KEYS = ["code", "func_code_string", "whole_func_string", "original_string"]
CODE_DOC_KEYS = ["docstring", "func_documentation_string"]
CODE_NAME_KEYS = ["func_name", "function_name"]


def _first(row: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    return default


def _load_alpaca(limit: int, seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    shuffled = ds.shuffle(buffer_size=max(1000, limit * 10), seed=seed)
    return [dict(row) for row in shuffled.take(limit)]


def _load_gsm8k(limit: int, seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    shuffled = ds.shuffle(buffer_size=max(1000, limit * 10), seed=seed)
    return [dict(row) for row in shuffled.take(limit)]


def _load_codexglue(limit: int, seed: int) -> List[Dict[str, Any]]:
    """CodeXGLUE code-to-text (python). The HF repo is script-only, so modern
    `datasets` releases cannot load it directly; read the auto-converted
    parquet branch instead and normalise columns."""
    import tempfile

    import pandas as pd
    from huggingface_hub import snapshot_download

    cache_dir = Path(REPO_ROOT) / "supplementary" / "data" / "_codexglue_cache"
    snapshot_download(
        repo_id="google/code_x_glue_ct_code_to_text",
        repo_type="dataset",
        revision="refs/convert/parquet",
        allow_patterns=["python/*"],
        local_dir=cache_dir,
    )
    frames = []
    for parquet_path in sorted(cache_dir.rglob("*.parquet")):
        if parquet_path.parent.name != "train":
            continue
        frame = pd.read_parquet(parquet_path)
        frames.append(frame)
    if not frames:
        raise RuntimeError("No CodeXGLUE train parquet files found in snapshot")
    frame = pd.concat(frames, ignore_index=True)
    rng = random.Random(seed)
    indices = rng.sample(range(len(frame)), min(limit * 3, len(frame)))
    rows: List[Dict[str, Any]] = []
    for idx in indices:
        raw = frame.iloc[idx].to_dict()
        rows.append(
            {
                "docstring": _first(raw, CODE_DOC_KEYS),
                "code": _first(raw, CODE_CODE_KEYS),
                "func_name": _first(raw, CODE_NAME_KEYS),
            }
        )
    # Guard against empty strings after normalisation, then trim to limit.
    rows = [row for row in rows if row["docstring"] or row["code"]]
    return rows[:limit]


LOADERS = {
    "alpaca_instruction": _load_alpaca,
    "gsm8k_math": _load_gsm8k,
    "codexglue_python": _load_codexglue,
}


def build_domain(name: str, out_dir: Path, train_limit: int, val_limit: int, shards: int, seed: int) -> Dict[str, Any]:
    spec = REAL_DATASET_SPECS[name]
    rows = LOADERS[name](train_limit + val_limit, seed + DOMAIN_KEYS.index(name))
    records = []
    for idx, row in enumerate(rows):
        text = spec.formatter(row)
        if not text.strip():
            continue
        split = "val" if idx < val_limit else "train"
        shard_idx = (idx - val_limit) % shards
        source_id = f"{spec.name}_shard_{shard_idx:02d}" if split == "train" else f"{spec.name}_validation"
        records.append(
            {
                "doc_id": f"{spec.name}-{split}-{idx:04d}",
                "source_id": source_id,
                "text": text,
                "label": 1,
                "domain": spec.domain,
                "split": split,
            }
        )
    out_path = out_dir / f"{spec.domain}.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    n_train = sum(1 for r in records if r["split"] == "train")
    n_val = sum(1 for r in records if r["split"] == "val")
    shard_sizes: Dict[str, int] = {}
    for r in records:
        if r["split"] == "train":
            shard_sizes[r["source_id"]] = shard_sizes.get(r["source_id"], 0) + 1
    stats = {
        "dataset": name,
        "domain": spec.domain,
        "train": n_train,
        "val": n_val,
        "shards": shard_sizes,
        "mean_chars": sum(len(r["text"]) for r in records) / max(1, len(records)),
    }
    print(json.dumps(stats, ensure_ascii=False))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--train-limit", type=int, default=TRAIN_LIMIT, help="train docs per domain (default 120)")
    parser.add_argument("--val-limit", type=int, default=VAL_LIMIT, help="val docs per domain (default 32)")
    parser.add_argument("--shards", type=int, default=SHARDS, help="source shards per domain (default 6)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_stats = [
        build_domain(name, args.out_dir, args.train_limit, args.val_limit, args.shards, args.seed)
        for name in DOMAIN_KEYS
    ]
    summary_path = args.out_dir / "build_summary.json"
    summary_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
