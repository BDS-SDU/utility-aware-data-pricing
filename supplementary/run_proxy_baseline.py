"""L0 proxy baseline on the supplementary corpora (paper's own code path).

Re-runs the paper's valuation pipeline -- the 256-dim hashed logistic
regression proxy with its classification-style value function (Eq. 9) -- on
exactly the same shard-partitioned corpora used by run_lm_ladder.py, so that
the proxy-level source ranking can be compared against real language-model
rankings at 70m/160m/410m scale.

Validation protocol mirrors the paper: in-domain positives plus OOD negatives
from the two competing domains (paper Section 4.1 used 6+3+3; the
supplementary corpora use 32 in-domain + 16+16 OOD).

Output: supplementary/results/proxy_baseline.csv with one row per source:
proxy_gain (Eq. 10), mean DQS, influence, Shapley, unified score, row count,
whitespace token count.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_pricing.data_types import DocumentRecord  # noqa: E402
from data_pricing.pipeline import DynamicDataValuationPipeline  # noqa: E402
from data_pricing.real_datasets import build_target_validation  # noqa: E402

HERE = Path(__file__).resolve().parent
NEGATIVES_PER_DOMAIN = 16


def load_domain(path: Path) -> Dict[str, List[DocumentRecord]]:
    train: List[DocumentRecord] = []
    val: List[DocumentRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            record = DocumentRecord(
                doc_id=row["doc_id"],
                source_id=row["source_id"],
                text=row["text"],
                label=row["label"],
                metadata={"domain": row["domain"], "split": row["split"]},
            )
            (val if row["split"] == "val" else train).append(record)
    return {"train": train, "val": val}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    parser.add_argument("--shapley-iterations", type=int, default=32)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    domain_data = {path.stem: load_domain(path) for path in sorted(args.data_dir.glob("*.jsonl"))}
    if not domain_data:
        raise SystemExit(f"No domain jsonl files under {args.data_dir}; run build_data.py first")

    rows = []
    for target_domain, splits in sorted(domain_data.items()):
        val_records = build_target_validation(
            target_domain=target_domain,
            domain_validation={d: s["val"] for d, s in domain_data.items()},
            negatives_per_domain=NEGATIVES_PER_DOMAIN,
        )
        pipeline = DynamicDataValuationPipeline()
        report = pipeline.run(splits["train"], val_records, shapley_iterations=args.shapley_iterations)
        for score in report.source_scores:
            rows.append(
                {
                    "domain": target_domain,
                    "source": score.source_id,
                    "proxy_gain": score.proxy_gain,
                    "dqs": score.mean_dqs,
                    "influence": score.influence_score,
                    "shapley": score.shapley_value,
                    "unified": score.unified_score,
                    "row_count": score.document_count,
                    "token_count_ws": sum(len(r.text.split()) for r in splits["train"] if r.source_id == score.source_id),
                }
            )
        top = max(report.source_scores, key=lambda s: s.proxy_gain)
        print(f"[{target_domain}] sources={len(report.source_scores)} top_proxy_gain={top.source_id}")

    out_path = args.out_dir / "proxy_baseline.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
