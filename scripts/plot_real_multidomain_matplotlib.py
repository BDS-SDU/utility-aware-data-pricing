from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[1]

METHOD_LABELS = {
    "row_count": "Rows",
    "token_count": "Tokens",
    "dqs_only": "DQS",
    "proxy_gain": "Proxy",
    "influence": "Influence",
    "shapley": "Shapley",
    "unified": "Unified",
    "calibrated_unified": "Calib.",
}

DOMAIN_LABELS = {
    "code_summarization": "Code",
    "general_instruction": "Instruction",
    "math_reasoning": "Math",
}

SOURCE_LABELS = {
    "alpaca_instruction": "Alpaca",
    "gsm8k_math": "GSM8K",
    "codexglue_python": "CodeXGLUE",
}

METHOD_ORDER = [
    "row_count",
    "token_count",
    "dqs_only",
    "proxy_gain",
    "influence",
    "shapley",
    "unified",
    "calibrated_unified",
]

PAPER_COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "black": "#222222",
    "gray": "#666666",
    "light_gray": "#D9D9D9",
}


def import_matplotlib():
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "This plotting script requires matplotlib and numpy. Install with:\n"
            "  pip install matplotlib numpy\n"
            f"Original import error: {exc}"
        ) from exc
    return mpl, plt, np


def default_report_path() -> Path:
    candidates = [
        ROOT / "outputs" / "real_multidomain" / "real_multidomain_report.json",
        ROOT / "outputs" / "real_multidomain_smoke" / "real_multidomain_report.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw real multi-domain experiment results with a SIGMOD/KDD-style matplotlib theme."
    )
    parser.add_argument("--report", default=str(default_report_path()))
    parser.add_argument("--outdir", default=str(ROOT / "outputs" / "figures"))
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        choices=["pdf", "png", "svg"],
        help="Output figure formats.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_report(path: str | Path) -> Dict[str, object]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Cannot find real multi-domain report: {target}")
    report = json.loads(target.read_text(encoding="utf-8"))
    if isinstance(report, str):
        report = json.loads(report)
    if not isinstance(report, dict):
        raise TypeError(f"Expected JSON object in {target}, got {type(report).__name__}.")
    return report


def parse_json_string(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def configure_style(mpl, plt) -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "axes.linewidth": 0.75,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": PAPER_COLORS["light_gray"],
            "grid.linewidth": 0.45,
            "grid.alpha": 0.8,
            "legend.frameon": False,
        }
    )
    plt.rcParams["hatch.linewidth"] = 0.6


def save_all(fig, outdir: Path, stem: str, formats: Sequence[str], dpi: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(outdir / f"{stem}.{fmt}", dpi=dpi)


def get_domain_results(report: Dict[str, object]) -> List[Dict[str, object]]:
    domain_results = parse_json_string(report.get("domain_results", []))
    if isinstance(domain_results, dict):
        domain_results = list(domain_results.values())
    if not isinstance(domain_results, list):
        raise TypeError(
            "Expected `domain_results` to be a list in real_multidomain_report.json, "
            f"got {type(domain_results).__name__}."
        )
    return [dict(row) for row in domain_results]


def get_ranking_metrics(row: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    metrics = parse_json_string(row.get("ranking_metrics", {}))
    if not isinstance(metrics, dict):
        raise TypeError(f"Expected `ranking_metrics` dict, got {type(metrics).__name__}.")
    return metrics  # type: ignore[return-value]


def get_actual_gains(row: Dict[str, object]) -> Dict[str, float]:
    gains = parse_json_string(row.get("actual_gains", {}))
    if not isinstance(gains, dict):
        raise TypeError(f"Expected `actual_gains` dict, got {type(gains).__name__}.")
    return {str(key): float(value) for key, value in gains.items()}


def mean_ranking_metrics_from_domains(domain_results: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    if not domain_results:
        return {}
    methods = sorted(get_ranking_metrics(domain_results[0]).keys())
    summary: Dict[str, Dict[str, float]] = {}
    for method in methods:
        rho_values: List[float] = []
        top2_values: List[float] = []
        for row in domain_results:
            metrics = get_ranking_metrics(row)
            if method not in metrics:
                continue
            rho_values.append(float(metrics[method].get("spearman_rho", 0.0)))
            top2_values.append(float(metrics[method].get("top2_overlap", 0.0)))
        if rho_values:
            summary[method] = {
                "mean_spearman_rho": sum(rho_values) / len(rho_values),
                "mean_top2_overlap": sum(top2_values) / len(top2_values),
            }
    return summary


def get_mean_ranking_metrics(report: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    summary = parse_json_string(report.get("summary", {}))
    if isinstance(summary, dict):
        metrics = parse_json_string(summary.get("mean_ranking_metrics", {}))
        if isinstance(metrics, dict) and metrics:
            return {
                str(method): {
                    "mean_spearman_rho": float(values.get("mean_spearman_rho", 0.0)),
                    "mean_top2_overlap": float(values.get("mean_top2_overlap", 0.0)),
                }
                for method, values in metrics.items()
                if isinstance(values, dict)
            }
    computed = mean_ranking_metrics_from_domains(get_domain_results(report))
    if not computed:
        raise TypeError("Cannot find or reconstruct summary.mean_ranking_metrics from the report.")
    return computed


def source_family(source_id: str) -> str:
    for prefix, label in SOURCE_LABELS.items():
        if source_id.startswith(prefix):
            return label
    return source_id.split("_shard_")[0]


def plot_summary_metrics(report: Dict[str, object], outdir: Path, formats: Sequence[str], dpi: int) -> None:
    _, plt, np = import_matplotlib()
    summary = get_mean_ranking_metrics(report)
    methods = [m for m in METHOD_ORDER if m in summary]
    x = np.arange(len(methods))
    width = 0.36
    rho = [summary[m]["mean_spearman_rho"] for m in methods]
    top2 = [summary[m]["mean_top2_overlap"] for m in methods]

    fig, ax = plt.subplots(figsize=(4.85, 2.35))
    ax.axhline(0.0, color=PAPER_COLORS["black"], linewidth=0.6, zorder=1)
    ax.bar(
        x - width / 2,
        rho,
        width,
        label=r"Mean Spearman $\rho$",
        color=PAPER_COLORS["blue"],
        edgecolor=PAPER_COLORS["black"],
        linewidth=0.45,
        hatch="//",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        top2,
        width,
        label="Mean Top-2 overlap",
        color=PAPER_COLORS["orange"],
        edgecolor=PAPER_COLORS["black"],
        linewidth=0.45,
        hatch="\\\\",
        zorder=3,
    )
    ax.set_ylabel("Score")
    ax.set_ylim(min(-1.0, min(rho + top2) - 0.05), 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=25, ha="right")
    # ax.set_title("Real Multi-Domain Ranking Quality")
    ax.legend(ncol=1, loc="upper left")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    save_all(fig, outdir, "real_summary_metrics", formats, dpi)
    plt.close(fig)


def plot_domain_rho(report: Dict[str, object], outdir: Path, formats: Sequence[str], dpi: int) -> None:
    _, plt, np = import_matplotlib()
    domains = get_domain_results(report)
    methods = ["row_count", "token_count", "dqs_only", "proxy_gain", "unified", "calibrated_unified"]
    colors = [
        PAPER_COLORS["gray"],
        PAPER_COLORS["orange"],
        PAPER_COLORS["green"],
        PAPER_COLORS["blue"],
        PAPER_COLORS["purple"],
        PAPER_COLORS["vermillion"],
    ]
    hatches = ["", "//", "\\\\", "..", "xx", "++"]
    x = np.arange(len(domains))
    width = 0.12

    fig, ax = plt.subplots(figsize=(4.85, 2.45))
    ax.axhline(0.0, color=PAPER_COLORS["black"], linewidth=0.6, zorder=1)
    for idx, method in enumerate(methods):
        values = [get_ranking_metrics(row)[method]["spearman_rho"] for row in domains]
        offset = (idx - (len(methods) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=METHOD_LABELS[method],
            color=colors[idx],
            edgecolor=PAPER_COLORS["black"],
            linewidth=0.4,
            hatch=hatches[idx],
            zorder=3,
        )
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABELS.get(row["target_domain"], row["target_domain"]) for row in domains])
    # ax.set_title("Per-Domain Rank Correlation")
    # ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0.0)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    save_all(fig, outdir, "real_domain_rho", formats, dpi)
    plt.close(fig)


def plot_method_heatmap(report: Dict[str, object], outdir: Path, formats: Sequence[str], dpi: int) -> None:
    _, plt, np = import_matplotlib()
    domains = get_domain_results(report)
    methods = ["row_count", "token_count", "dqs_only", "proxy_gain", "influence", "shapley", "unified", "calibrated_unified"]
    data = np.array(
        [
            [get_ranking_metrics(row)[method]["spearman_rho"] for method in methods]
            for row in domains
        ]
    )
    fig, ax = plt.subplots(figsize=(4.9, 1.95))
    im = ax.imshow(data, cmap="RdBu", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(domains)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=30, ha="right")
    ax.set_yticklabels([DOMAIN_LABELS.get(row["target_domain"], row["target_domain"]) for row in domains])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = "white" if abs(data[i, j]) > 0.6 else PAPER_COLORS["black"]
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color=color)
    # ax.set_title(r"Spearman $\rho$ by Target Domain and Estimator")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(r"$\rho$", rotation=0, labelpad=7)
    save_all(fig, outdir, "real_method_heatmap", formats, dpi)
    plt.close(fig)


def plot_top_source_domains(report: Dict[str, object], outdir: Path, formats: Sequence[str], dpi: int) -> None:
    _, plt, np = import_matplotlib()
    domains = get_domain_results(report)
    source_families = ["Alpaca", "GSM8K", "CodeXGLUE"]
    matrix = np.zeros((len(domains), len(source_families)))
    for row_idx, row in enumerate(domains):
        gains = get_actual_gains(row)
        for source_id, gain in gains.items():
            family = source_family(source_id)
            if family in source_families:
                matrix[row_idx, source_families.index(family)] = max(matrix[row_idx, source_families.index(family)], float(gain))

    fig, ax = plt.subplots(figsize=(3.6, 2.15))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(source_families)))
    ax.set_yticks(np.arange(len(domains)))
    ax.set_xticklabels(source_families)
    ax.set_yticklabels([DOMAIN_LABELS.get(row["target_domain"], row["target_domain"]) for row in domains])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center", fontsize=6.5)
    # ax.set_title("Best Realized Gain by Source Family")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label(r"$G_i$", rotation=0, labelpad=8)
    save_all(fig, outdir, "real_top_source_domains", formats, dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    mpl, plt, _ = import_matplotlib()
    configure_style(mpl, plt)
    report = load_report(args.report)
    outdir = Path(args.outdir)

    plot_summary_metrics(report, outdir, args.formats, args.dpi)
    plot_domain_rho(report, outdir, args.formats, args.dpi)
    plot_method_heatmap(report, outdir, args.formats, args.dpi)
    plot_top_source_domains(report, outdir, args.formats, args.dpi)
    print(f"Saved real multi-domain figures to {outdir}")
    for stem in [
        "real_summary_metrics",
        "real_domain_rho",
        "real_method_heatmap",
        "real_top_source_domains",
    ]:
        print("  " + ", ".join(str(outdir / f"{stem}.{fmt}") for fmt in args.formats))


if __name__ == "__main__":
    main()
