from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]

PAPER_COLORS = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "black": "#222222",
    "gray": "#666666",
    "light_gray": "#E6E6E6",
    "panel": "#FFFFFF",
}

DISPLAY_LABELS = {
    "row_count": "Rows",
    "token_count": "Tokens",
    "dqs_only": "DQS",
    "proxy_gain": "Proxy",
    "unified": "Unified",
    "calibrated_unified": "Calib.",
    "api_gold": "API",
    "reasoning_gold": "Reasoning",
    "faq_mixed": "FAQ",
    "redundant_copy": "Redundant",
    "noise_dump": "Noise",
    "general_instruction": "Instruction",
    "math_reasoning": "Math",
    "code_summarization": "Code",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot saved experiment and valuation results to SVG.")
    parser.add_argument("--experiment-json", default=str(ROOT / "outputs" / "experiment_report.json"))
    parser.add_argument("--valuation-json", default=str(ROOT / "outputs" / "valuation_report.json"))
    parser.add_argument("--real-json", default=str(ROOT / "outputs" / "real_multidomain" / "real_multidomain_report.json"))
    parser.add_argument("--outdir", default=str(ROOT / "outputs" / "figures"))
    return parser.parse_args()


def load_json(path: str | Path) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        "text { font-family: 'Times New Roman', Times, serif; fill: #222222; }",
        ".title { font-size: 17px; font-weight: bold; }",
        ".label { font-size: 11px; }",
        ".tick { font-size: 10px; }",
        ".legend { font-size: 11px; }",
        ".value { font-size: 9px; fill: #333333; }",
        ".axis-title { font-size: 12px; font-weight: bold; }",
        "</style>",
    ]


def save_svg(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines + ["</svg>"]))


def grouped_bar_chart(
    title: str,
    labels: List[str],
    series: List[Dict[str, object]],
    y_min: float,
    y_max: float,
    output_path: Path,
    y_label: str,
) -> None:
    width = 820
    height = 470
    left = 78
    right = 28
    top = 58
    bottom = 96
    plot_w = width - left - right
    plot_h = height - top - bottom
    lines = svg_header(width, height)
    lines.append(f'<text x="{left}" y="32" class="title">{title}</text>')
    lines.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#FFFFFF"/>')
    lines.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="{PAPER_COLORS["panel"]}" stroke="{PAPER_COLORS["black"]}" stroke-width="1.1"/>')
    lines.append(
        f'<text x="20" y="{top + plot_h / 2:.1f}" class="axis-title" transform="rotate(-90 20 {top + plot_h / 2:.1f})" text-anchor="middle">{y_label}</text>'
    )

    tick_count = 5
    for idx in range(tick_count + 1):
        ratio = idx / tick_count
        value = y_min + (y_max - y_min) * (1 - ratio)
        y = top + plot_h * ratio
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{PAPER_COLORS["light_gray"]}" stroke-width="0.8"/>')
        lines.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="{PAPER_COLORS["black"]}" stroke-width="1"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.2f}</text>')

    group_width = plot_w / max(1, len(labels))
    inner_padding = 16
    series_count = max(1, len(series))
    bar_width = max(8.0, (group_width - inner_padding * 2) / series_count)
    baseline_value = 0.0
    baseline_y = top + plot_h * (1 - (baseline_value - y_min) / max(1e-12, y_max - y_min))
    if y_min < 0 < y_max:
        lines.append(f'<line x1="{left}" y1="{baseline_y:.1f}" x2="{left + plot_w}" y2="{baseline_y:.1f}" stroke="{PAPER_COLORS["gray"]}" stroke-width="1" stroke-dasharray="4 4"/>')

    for label_idx, label in enumerate(labels):
        x0 = left + label_idx * group_width + inner_padding
        display_label = DISPLAY_LABELS.get(label, label)
        lines.append(
            f'<text x="{left + label_idx * group_width + group_width / 2:.1f}" y="{height - 58}" text-anchor="middle" class="label">{display_label}</text>'
        )
        for series_idx, spec in enumerate(series):
            value = float(spec["values"][label_idx])
            color = str(spec["color"])
            x = x0 + series_idx * bar_width
            y_value = top + plot_h * (1 - (value - y_min) / max(1e-12, y_max - y_min))
            y = min(y_value, baseline_y)
            bar_h = abs(baseline_y - y_value)
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 4:.1f}" height="{bar_h:.1f}" fill="{color}" stroke="#222222" stroke-width="0.45"/>'
            )
            if bar_width >= 18:
                value_y = y - 6 if y > top + 18 else y + 14
                lines.append(
                    f'<text x="{x + (bar_width - 4) / 2:.1f}" y="{value_y:.1f}" text-anchor="middle" class="value">{value:.2f}</text>'
                )

    legend_x = left
    legend_y = height - 22
    for idx, spec in enumerate(series):
        x = legend_x + idx * 142
        lines.append(f'<rect x="{x}" y="{legend_y - 10}" width="12" height="12" fill="{spec["color"]}" stroke="#222222" stroke-width="0.45"/>')
        lines.append(f'<text x="{x + 18}" y="{legend_y}" class="legend">{spec["name"]}</text>')
    save_svg(output_path, lines)


def plot_ranking_metrics(experiment_payload: Dict[str, object], output_dir: Path) -> None:
    labels = list(experiment_payload["ranking_metrics"].keys())
    rho_values = [experiment_payload["ranking_metrics"][label]["spearman_rho"] for label in labels]
    top2_values = [experiment_payload["ranking_metrics"][label]["top2_overlap"] for label in labels]
    grouped_bar_chart(
        title="Ranking Metrics by Estimator",
        labels=labels,
        series=[
            {"name": "Spearman rho", "values": rho_values, "color": PAPER_COLORS["blue"]},
            {"name": "Top-2 overlap", "values": top2_values, "color": PAPER_COLORS["orange"]},
        ],
        y_min=0.0,
        y_max=1.0,
        output_path=output_dir / "ranking_metrics.svg",
        y_label="Score",
    )


def plot_source_estimators(experiment_payload: Dict[str, object], output_dir: Path) -> None:
    labels = sorted(experiment_payload["actual_gains"].keys())
    actual = [experiment_payload["actual_gains"][label] for label in labels]
    proxy = [experiment_payload["estimators_by_source"]["proxy_gain"][label] for label in labels]
    calibrated = [experiment_payload["estimators_by_source"]["calibrated_unified"][label] for label in labels]
    grouped_bar_chart(
        title="Actual Gain vs Estimated Source Value",
        labels=labels,
        series=[
            {"name": "Actual gain", "values": actual, "color": PAPER_COLORS["black"]},
            {"name": "Proxy gain", "values": proxy, "color": PAPER_COLORS["blue"]},
            {"name": "Calibrated unified", "values": calibrated, "color": PAPER_COLORS["vermillion"]},
        ],
        y_min=min(actual + proxy + calibrated) - 0.02,
        y_max=max(actual + proxy + calibrated) + 0.02,
        output_path=output_dir / "source_estimators.svg",
        y_label="Value",
    )


def plot_valuation_scores(valuation_payload: Dict[str, object], output_dir: Path) -> None:
    source_scores = valuation_payload["source_scores"]
    labels = [item["source_id"] for item in source_scores]
    unified = [item["unified_score"] for item in source_scores]
    dqs = [item["mean_dqs"] for item in source_scores]
    proxy = [item["proxy_gain"] for item in source_scores]
    grouped_bar_chart(
        title="Valuation Components by Source",
        labels=labels,
        series=[
            {"name": "Unified", "values": unified, "color": PAPER_COLORS["blue"]},
            {"name": "Mean DQS", "values": dqs, "color": PAPER_COLORS["green"]},
            {"name": "Proxy gain", "values": proxy, "color": PAPER_COLORS["vermillion"]},
        ],
        y_min=min(proxy + [0.0]),
        y_max=max(unified + dqs + [0.1]),
        output_path=output_dir / "valuation_components.svg",
        y_label="Normalized value",
    )


def plot_real_multidomain_summary(real_payload: Dict[str, object], output_dir: Path) -> None:
    summary = real_payload["summary"]["mean_ranking_metrics"]
    method_order = [
        "row_count",
        "token_count",
        "dqs_only",
        "proxy_gain",
        "unified",
        "calibrated_unified",
    ]
    labels = [method for method in method_order if method in summary]
    rho = [summary[label]["mean_spearman_rho"] for label in labels]
    top2 = [summary[label]["mean_top2_overlap"] for label in labels]
    grouped_bar_chart(
        title="Real Multi-Domain Mean Ranking Metrics",
        labels=labels,
        series=[
            {"name": "Mean Spearman rho", "values": rho, "color": PAPER_COLORS["blue"]},
            {"name": "Mean Top-2 overlap", "values": top2, "color": PAPER_COLORS["orange"]},
        ],
        y_min=min(0.0, min(rho + top2)),
        y_max=1.0,
        output_path=output_dir / "real_multidomain_summary.svg",
        y_label="Score",
    )


def plot_real_domain_rho(real_payload: Dict[str, object], output_dir: Path) -> None:
    domain_results = real_payload["domain_results"]
    labels = [row["target_domain"] for row in domain_results]
    series = []
    for method, color in [
        ("row_count", PAPER_COLORS["gray"]),
        ("token_count", PAPER_COLORS["orange"]),
        ("proxy_gain", PAPER_COLORS["blue"]),
        ("calibrated_unified", PAPER_COLORS["vermillion"]),
    ]:
        values = [row["ranking_metrics"][method]["spearman_rho"] for row in domain_results]
        series.append({"name": DISPLAY_LABELS.get(method, method), "values": values, "color": color})
    all_values = [value for spec in series for value in spec["values"]]
    grouped_bar_chart(
        title="Per-Domain Rank Correlation on Real Datasets",
        labels=labels,
        series=series,
        y_min=min(-1.0, min(all_values)),
        y_max=1.0,
        output_path=output_dir / "real_domain_rho.svg",
        y_label="Spearman rho",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.outdir)
    experiment_payload = load_json(args.experiment_json)
    valuation_payload = load_json(args.valuation_json)
    plot_ranking_metrics(experiment_payload, output_dir)
    plot_source_estimators(experiment_payload, output_dir)
    plot_valuation_scores(valuation_payload, output_dir)
    real_json = Path(args.real_json)
    if real_json.exists():
        real_payload = load_json(real_json)
        plot_real_multidomain_summary(real_payload, output_dir)
        plot_real_domain_rho(real_payload, output_dir)
    print(f"Saved figures to {output_dir}")
    print(f"  {output_dir / 'ranking_metrics.svg'}")
    print(f"  {output_dir / 'source_estimators.svg'}")
    print(f"  {output_dir / 'valuation_components.svg'}")
    if Path(args.real_json).exists():
        print(f"  {output_dir / 'real_multidomain_summary.svg'}")
        print(f"  {output_dir / 'real_domain_rho.svg'}")


if __name__ == "__main__":
    main()
