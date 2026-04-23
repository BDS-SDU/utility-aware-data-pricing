from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pricing.demo_data import build_demo_rows
from data_pricing.io_utils import write_jsonl


def main() -> None:
    train_rows, val_rows = build_demo_rows()
    write_jsonl(ROOT / "data" / "demo_train.jsonl", train_rows)
    write_jsonl(ROOT / "data" / "demo_val.jsonl", val_rows)
    print("Wrote demo data to data/demo_train.jsonl and data/demo_val.jsonl")


if __name__ == "__main__":
    main()
