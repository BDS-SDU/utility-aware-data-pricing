from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pricing.io_utils import read_jsonl, write_csv, write_json
from data_pricing.pipeline import DynamicDataValuationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the dynamic data valuation pipeline.")
    parser.add_argument("--train", default=str(ROOT / "data" / "demo_train.jsonl"))
    parser.add_argument("--val", default=str(ROOT / "data" / "demo_val.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "valuation_report.json"))
    parser.add_argument("--shapley-iterations", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_records = read_jsonl(args.train)
    val_records = read_jsonl(args.val)
    pipeline = DynamicDataValuationPipeline()
    report = pipeline.run(train_records, val_records, shapley_iterations=args.shapley_iterations)
    output_path = Path(args.output)
    tables_dir = output_path.parent / "tables"
    write_json(output_path, report.to_dict())
    write_csv(
        tables_dir / "valuation_source_scores.csv",
        [
            "source_id",
            "document_count",
            "token_count",
            "mean_dqs",
            "proxy_gain",
            "influence_score",
            "shapley_value",
            "unified_score",
            "confidence_low",
            "confidence_high",
        ],
        [source.to_dict() for source in report.source_scores],
    )
    write_csv(
        tables_dir / "valuation_document_scores.csv",
        [
            "doc_id",
            "source_id",
            "token_count",
            "info_density_bits",
            "normalized_info_density",
            "syntactic_coherence",
            "semantic_richness",
            "dqs",
        ],
        [doc.to_dict() for doc in report.document_scores],
    )
    print(f"Saved valuation report to {args.output}")
    print(f"Saved valuation tables to {tables_dir}")
    print("Top ranked sources:")
    for source in report.source_scores[:5]:
        print(
            f"  {source.source_id:16s} unified={source.unified_score:.4f} "
            f"proxy={source.proxy_gain:.4f} shapley={source.shapley_value:.4f}"
        )


if __name__ == "__main__":
    main()
