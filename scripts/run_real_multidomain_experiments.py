from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pricing.data_types import DocumentRecord
from data_pricing.io_utils import write_csv, write_json
from data_pricing.pipeline import DynamicDataValuationPipeline
from data_pricing.real_datasets import (
    REAL_DATASET_SPECS,
    build_target_validation,
    load_all_real_domains,
)
from data_pricing.valuation import ProxyEvaluator

from run_experiments import combine_scores, learn_weights, spearman_rho, top_k_overlap


def default_local_data_dir(dataset_names: List[str]) -> str | None:
    data_dir = ROOT / "data"
    if not data_dir.exists():
        return None
    expected = [data_dir / f"{dataset_name}.jsonl" for dataset_name in dataset_names]
    if all(path.exists() for path in expected):
        return str(data_dir)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified valuation analysis on real multi-domain fine-tuning datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["alpaca_instruction", "gsm8k_math", "codexglue_python"],
        choices=sorted(REAL_DATASET_SPECS.keys()),
    )
    parser.add_argument("--train-limit-per-domain", type=int, default=120)
    parser.add_argument("--val-limit-per-domain", type=int, default=40)
    parser.add_argument("--shards-per-domain", type=int, default=4)
    parser.add_argument("--negative-val-per-domain", type=int, default=20)
    parser.add_argument("--shapley-iterations", type=int, default=32)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "real_multidomain"))
    parser.add_argument(
        "--local-data-dir",
        default=None,
        help=(
            "Optional offline JSONL directory. Expected files include "
            "alpaca_instruction.jsonl, gsm8k_math.jsonl, and codexglue_python.jsonl."
        ),
    )
    parser.add_argument(
        "--hf-local-files-only",
        action="store_true",
        help="Use only datasets already available in the local Hugging Face cache.",
    )
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def stronger_actual_gains(
    train_records: List[DocumentRecord],
    val_records: List[DocumentRecord],
) -> Dict[str, float]:
    evaluator = ProxyEvaluator(dim=512, lr=0.10, epochs=20, l2=5e-4, proxy_scale=1_300_000_000.0)
    return evaluator.source_gains(train_records, val_records)


def source_domain(source_id: str) -> str:
    if source_id.startswith("alpaca_instruction"):
        return "general_instruction"
    if source_id.startswith("gsm8k_math"):
        return "math_reasoning"
    if source_id.startswith("codexglue_python"):
        return "code_summarization"
    return "unknown"


def aggregate_estimators(report) -> Dict[str, Dict[str, float]]:
    estimators = {
        "dqs_only": {},
        "proxy_gain": {},
        "influence": {},
        "shapley": {},
        "unified": {},
    }
    for source in report.source_scores:
        estimators["dqs_only"][source.source_id] = source.mean_dqs
        estimators["proxy_gain"][source.source_id] = source.proxy_gain
        estimators["influence"][source.source_id] = source.influence_score
        estimators["shapley"][source.source_id] = source.shapley_value
        estimators["unified"][source.source_id] = source.unified_score
    return estimators


def source_counts(train_records: List[DocumentRecord]) -> Dict[str, Dict[str, float]]:
    row_count: Dict[str, float] = {}
    token_count: Dict[str, float] = {}
    for record in train_records:
        row_count[record.source_id] = row_count.get(record.source_id, 0.0) + 1.0
        token_count[record.source_id] = token_count.get(record.source_id, 0.0) + float(len(record.text.split()))
    return {"row_count": row_count, "token_count": token_count}


def save_domain_tables(output_dir: Path, payload: Dict[str, object]) -> None:
    tables_dir = output_dir / "tables"
    ranking_rows = []
    for row in payload["domain_results"]:
        target_domain = row["target_domain"]
        for method, metrics in row["ranking_metrics"].items():
            ranking_rows.append(
                {
                    "target_domain": target_domain,
                    "method": method,
                    "spearman_rho": metrics["spearman_rho"],
                    "top2_overlap": metrics["top2_overlap"],
                }
            )
    write_csv(
        tables_dir / "domain_ranking_metrics.csv",
        ["target_domain", "method", "spearman_rho", "top2_overlap"],
        ranking_rows,
    )

    source_rows = []
    for row in payload["domain_results"]:
        target_domain = row["target_domain"]
        actual = row["actual_gains"]
        estimators = row["estimators_by_source"]
        for source_id in sorted(actual.keys()):
            source_row = {
                "target_domain": target_domain,
                "source_id": source_id,
                "source_domain": source_domain(source_id),
                "actual_gain": actual[source_id],
            }
            for name, values in estimators.items():
                source_row[name] = values[source_id]
            source_rows.append(source_row)
    write_csv(
        tables_dir / "domain_source_estimators.csv",
        [
            "target_domain",
            "source_id",
            "source_domain",
            "actual_gain",
            "row_count",
            "token_count",
            "dqs_only",
            "proxy_gain",
            "influence",
            "shapley",
            "unified",
            "calibrated_unified",
        ],
        source_rows,
    )

    summary_rows = []
    for method, metrics in payload["summary"]["mean_ranking_metrics"].items():
        summary_rows.append(
            {
                "method": method,
                "mean_spearman_rho": metrics["mean_spearman_rho"],
                "mean_top2_overlap": metrics["mean_top2_overlap"],
            }
        )
    write_csv(
        tables_dir / "summary_ranking_metrics.csv",
        ["method", "mean_spearman_rho", "mean_top2_overlap"],
        summary_rows,
    )


def mean_metrics(domain_results: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    methods = sorted(domain_results[0]["ranking_metrics"].keys())
    summary: Dict[str, Dict[str, float]] = {}
    for method in methods:
        rho_values = [row["ranking_metrics"][method]["spearman_rho"] for row in domain_results]
        top2_values = [row["ranking_metrics"][method]["top2_overlap"] for row in domain_results]
        summary[method] = {
            "mean_spearman_rho": sum(rho_values) / len(rho_values),
            "mean_top2_overlap": sum(top2_values) / len(top2_values),
        }
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    local_data_dir = args.local_data_dir or default_local_data_dir(args.datasets)
    if local_data_dir:
        print(f"Using local JSONL datasets from {local_data_dir}")
    elif args.hf_local_files_only:
        print("Using local Hugging Face cache only")
    else:
        print("No complete local JSONL set detected; loading from Hugging Face")
    domain_data = load_all_real_domains(
        dataset_names=args.datasets,
        train_limit_per_domain=args.train_limit_per_domain,
        val_limit_per_domain=args.val_limit_per_domain,
        shards_per_domain=args.shards_per_domain,
        seed=args.seed,
        local_data_dir=local_data_dir,
        hf_local_files_only=args.hf_local_files_only,
    )
    train_records: List[DocumentRecord] = []
    domain_validation: Dict[str, List[DocumentRecord]] = {}
    for domain, splits in domain_data.items():
        train_records.extend(splits["train"])
        domain_validation[domain] = splits["val"]

    domain_results: List[Dict[str, object]] = []
    for target_domain in sorted(domain_validation.keys()):
        val_records = build_target_validation(
            target_domain=target_domain,
            domain_validation=domain_validation,
            negatives_per_domain=args.negative_val_per_domain,
        )
        pipeline = DynamicDataValuationPipeline()
        report = pipeline.run(train_records, val_records, shapley_iterations=args.shapley_iterations)
        actual = stronger_actual_gains(train_records, val_records)
        estimators = {**source_counts(train_records), **aggregate_estimators(report)}
        calibrated_components = {
            "dqs_only": estimators["dqs_only"],
            "proxy_gain": estimators["proxy_gain"],
            "influence": estimators["influence"],
            "shapley": estimators["shapley"],
        }
        learned_weights = learn_weights(calibrated_components, actual)
        estimators["calibrated_unified"] = combine_scores(calibrated_components, learned_weights)
        ranking_metrics = {
            name: {
                "spearman_rho": spearman_rho(values, actual),
                "top2_overlap": top_k_overlap(values, actual, k=2),
            }
            for name, values in estimators.items()
        }
        domain_results.append(
            {
                "target_domain": target_domain,
                "validation_size": len(val_records),
                "actual_gains": actual,
                "estimators_by_source": estimators,
                "learned_weights": learned_weights,
                "ranking_metrics": ranking_metrics,
                "top_sources": [source.to_dict() for source in report.source_scores[:10]],
                "ledger": report.ledger,
            }
        )
        print(f"[{target_domain}] calibrated rho={ranking_metrics['calibrated_unified']['spearman_rho']:.3f}")

    payload = {
        "dataset_specs": {
            name: {
                "domain": REAL_DATASET_SPECS[name].domain,
                "hf_path": REAL_DATASET_SPECS[name].hf_path,
                "hf_config": REAL_DATASET_SPECS[name].hf_config,
                "split": REAL_DATASET_SPECS[name].split,
                "citation_url": REAL_DATASET_SPECS[name].citation_url,
                "license_note": REAL_DATASET_SPECS[name].license_note,
            }
            for name in args.datasets
        },
        "config": vars(args),
        "domain_results": domain_results,
        "summary": {"mean_ranking_metrics": mean_metrics(domain_results)},
    }
    write_json(output_dir / "real_multidomain_report.json", payload)
    save_domain_tables(output_dir, payload)
    print(f"Saved real multi-domain report to {output_dir / 'real_multidomain_report.json'}")
    print(f"Saved real multi-domain tables to {output_dir / 'tables'}")


if __name__ == "__main__":
    main()
