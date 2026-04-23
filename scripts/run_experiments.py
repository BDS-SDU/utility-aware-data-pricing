from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pricing.data_types import DocumentRecord
from data_pricing.demo_data import build_demo_rows
from data_pricing.io_utils import write_csv, write_json
from data_pricing.pipeline import DynamicDataValuationPipeline
from data_pricing.valuation import ProxyEvaluator


def rank_map(values: Dict[str, float]) -> Dict[str, int]:
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=True)
    return {key: idx + 1 for idx, (key, _) in enumerate(ordered)}


def spearman_rho(left: Dict[str, float], right: Dict[str, float]) -> float:
    common = sorted(set(left) & set(right))
    if len(common) <= 1:
        return 0.0
    left_ranks = rank_map({key: left[key] for key in common})
    right_ranks = rank_map({key: right[key] for key in common})
    numerator = 6.0 * sum((left_ranks[key] - right_ranks[key]) ** 2 for key in common)
    denominator = len(common) * (len(common) ** 2 - 1)
    return 1.0 - numerator / denominator


def top_k_overlap(left: Dict[str, float], right: Dict[str, float], k: int = 2) -> float:
    left_top = {key for key, _ in sorted(left.items(), key=lambda item: item[1], reverse=True)[:k]}
    right_top = {key for key, _ in sorted(right.items(), key=lambda item: item[1], reverse=True)[:k]}
    return len(left_top & right_top) / max(1, k)


def rows_to_records(rows: Iterable[dict]) -> List[DocumentRecord]:
    return [DocumentRecord.from_dict(row) for row in rows]


def learn_weights(components: Dict[str, Dict[str, float]], target: Dict[str, float]) -> Dict[str, float]:
    raw = {}
    for name, values in components.items():
        correlation = spearman_rho(values, target)
        raw[name] = max(0.0, correlation)
    total = sum(raw.values())
    if total == 0.0:
        return {name: 1.0 / len(raw) for name in raw}
    return {name: value / total for name, value in raw.items()}


def combine_scores(components: Dict[str, Dict[str, float]], weights: Dict[str, float]) -> Dict[str, float]:
    source_ids = sorted(next(iter(components.values())).keys())
    combined = {}
    for source_id in source_ids:
        combined[source_id] = sum(weights[name] * components[name][source_id] for name in components)
    return combined


def stronger_actual_gains(
    train_records: Sequence[DocumentRecord],
    val_records: Sequence[DocumentRecord],
) -> Dict[str, float]:
    evaluator = ProxyEvaluator(dim=384, lr=0.12, epochs=18, l2=5e-4, proxy_scale=1_300_000_000.0)
    return evaluator.source_gains(train_records, val_records)


def aggregate_baselines(train_records: Sequence[DocumentRecord], report) -> Dict[str, Dict[str, float]]:
    grouped_tokens = defaultdict(int)
    grouped_docs = defaultdict(int)
    for record in train_records:
        grouped_docs[record.source_id] += 1
        grouped_tokens[record.source_id] += len(record.text.split())
    dqs = {item.source_id: item.mean_dqs for item in report.source_scores}
    proxy = {item.source_id: item.proxy_gain for item in report.source_scores}
    unified = {item.source_id: item.unified_score for item in report.source_scores}
    return {
        "row_count": dict(grouped_docs),
        "token_count": dict(grouped_tokens),
        "dqs_only": dqs,
        "proxy_gain": proxy,
        "unified": unified,
    }


def duplicate_noise_rows(train_rows: List[dict], copies: int = 5) -> List[dict]:
    extra: List[dict] = []
    for row in train_rows:
        if row["source_id"] != "noise_dump":
            continue
        for idx in range(copies):
            dup = dict(row)
            dup["doc_id"] = f"{row['doc_id']}-dup-{idx + 1}"
            extra.append(dup)
    return train_rows + extra


def save_tables(output_dir: Path, payload: Dict[str, object]) -> None:
    tables_dir = output_dir / "tables"
    ranking_rows = []
    for method, metrics in payload["ranking_metrics"].items():
        ranking_rows.append(
            {
                "method": method,
                "spearman_rho": metrics["spearman_rho"],
                "top2_overlap": metrics["top2_overlap"],
            }
        )
    write_csv(tables_dir / "ranking_metrics.csv", ["method", "spearman_rho", "top2_overlap"], ranking_rows)

    source_rows = []
    estimators = payload["estimators_by_source"]
    actual = payload["actual_gains"]
    all_sources = sorted(actual.keys())
    for source_id in all_sources:
        row = {"source_id": source_id, "actual_gain": actual[source_id]}
        for estimator_name, values in estimators.items():
            row[estimator_name] = values[source_id]
        source_rows.append(row)
    write_csv(
        tables_dir / "source_estimators.csv",
        ["source_id", "actual_gain"] + list(estimators.keys()),
        source_rows,
    )

    weight_rows = [
        {"component": component, "weight": weight}
        for component, weight in payload["learned_weights"].items()
    ]
    write_csv(tables_dir / "learned_weights.csv", ["component", "weight"], weight_rows)

    robustness_rows = []
    for method, values in payload["robustness"].items():
        robustness_rows.append(
            {
                "method": method,
                "clean_top1": values["clean_top1"],
                "attack_top1": values["attack_top1"],
            }
        )
    write_csv(tables_dir / "robustness.csv", ["method", "clean_top1", "attack_top1"], robustness_rows)

    ablation_rows = [
        {"ablation": name, "spearman_rho": metrics["spearman_rho"]}
        for name, metrics in payload["ablations"].items()
    ]
    write_csv(tables_dir / "ablations.csv", ["ablation", "spearman_rho"], ablation_rows)


def run() -> Dict[str, object]:
    train_rows, val_rows = build_demo_rows()
    train_records = rows_to_records(train_rows)
    val_records = rows_to_records(val_rows)

    baseline_pipeline = DynamicDataValuationPipeline()
    report = baseline_pipeline.run(train_records, val_records, shapley_iterations=80)
    actual = stronger_actual_gains(train_records, val_records)
    baselines = aggregate_baselines(train_records, report)
    calibrated_components = {
        "dqs_only": baselines["dqs_only"],
        "proxy_gain": baselines["proxy_gain"],
        "influence": {item.source_id: item.influence_score for item in report.source_scores},
        "shapley": {item.source_id: item.shapley_value for item in report.source_scores},
    }
    learned_weights = learn_weights(calibrated_components, actual)
    baselines["calibrated_unified"] = combine_scores(calibrated_components, learned_weights)

    ranking_metrics = {}
    for name, estimator in baselines.items():
        ranking_metrics[name] = {
            "spearman_rho": spearman_rho(estimator, actual),
            "top2_overlap": top_k_overlap(estimator, actual, k=2),
        }

    attack_rows = duplicate_noise_rows(train_rows, copies=6)
    attack_records = rows_to_records(attack_rows)
    attacked_report = baseline_pipeline.run(attack_records, val_records, shapley_iterations=80)
    attacked_baselines = aggregate_baselines(attack_records, attacked_report)
    robustness = {}
    for name in ("row_count", "token_count", "unified"):
        robustness[name] = {
            "clean_top1": max(baselines[name], key=baselines[name].get),
            "attack_top1": max(attacked_baselines[name], key=attacked_baselines[name].get),
        }

    ablations = {}
    ablation_weights = {
        "remove_dqs": (0.0, 0.5, 0.25, 0.25),
        "remove_proxy": (0.35, 0.0, 0.325, 0.325),
        "remove_influence": (0.3, 0.45, 0.0, 0.25),
        "remove_shapley": (0.3, 0.45, 0.25, 0.0),
    }
    for name, weights in ablation_weights.items():
        ablation_pipeline = DynamicDataValuationPipeline(ensemble_weights=weights)
        ablation_report = ablation_pipeline.run(train_records, val_records, shapley_iterations=80)
        ablation_scores = {item.source_id: item.unified_score for item in ablation_report.source_scores}
        ablations[name] = {"spearman_rho": spearman_rho(ablation_scores, actual)}

    return {
        "ranking_metrics": ranking_metrics,
        "actual_gains": actual,
        "learned_weights": learned_weights,
        "estimators_by_source": baselines,
        "robustness": robustness,
        "ablations": ablations,
        "top_sources": [item.to_dict() for item in report.source_scores[:5]],
    }


def main() -> None:
    payload = run()
    output_dir = ROOT / "outputs"
    output = output_dir / "experiment_report.json"
    write_json(output, payload)
    save_tables(output_dir, payload)
    print(f"Saved experiment report to {output}")
    print(f"Saved experiment tables to {output_dir / 'tables'}")
    for name, metrics in payload["ranking_metrics"].items():
        print(f"{name:12s} rho={metrics['spearman_rho']:.3f} top2={metrics['top2_overlap']:.3f}")


if __name__ == "__main__":
    main()
