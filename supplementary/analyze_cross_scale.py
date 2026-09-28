"""Analysis for the cross-scale supplementary experiment.

Reads:
    results/proxy_baseline.csv      (L0: paper proxy signals on same corpora)
    results/lm_gains_pythia-*.csv   (L1-L3: LOSO NLL gains of real LMs)
    results/lm_detail_*.json        (per-run details incl. per-doc NLL)

Produces:
    results/cross_scale_matrix.csv       pooled Spearman between all methods
    results/ranking_vs_410m.csv          per-domain + mean ranking agreement
    results/gain_magnitudes.csv          |gain| per domain x scale + bootstrap CI
    results/alpha_calibration.csv        empirical scaling exponent per source
    results/top_sources.csv              top-1 source per domain per scale
    results/summary.json                 every headline number
    figures/*.pdf + *.png                publication figures
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

LM_SCALES = ["pythia-70m", "pythia-160m", "pythia-410m"]
LM_PARAMS = {"pythia-70m": 70_426_624, "pythia-160m": 162_322_944, "pythia-410m": 405_334_016}
REFERENCE = "pythia-410m"

PROXY_SIGNALS = ["proxy_gain", "dqs", "influence", "shapley", "unified", "row_count", "token_count_ws"]


def load_tables() -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    proxy = pd.read_csv(RESULTS / "proxy_baseline.csv")
    proxy["scale"] = "proxy_logreg"
    tables["proxy"] = proxy
    for scale in LM_SCALES:
        frame = pd.read_csv(RESULTS / f"lm_gains_{scale}.csv")
        frame["scale"] = scale
        tables[scale] = frame
    return tables


def gains_wide(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (domain, source) with gains at every scale + proxy signals."""
    lm_frames = []
    for scale in LM_SCALES:
        frame = tables[scale][["domain", "source", "gain"]].rename(columns={"gain": scale})
        lm_frames.append(frame)
    wide = lm_frames[0]
    for frame in lm_frames[1:]:
        wide = wide.merge(frame, on=["domain", "source"])
    proxy = tables["proxy"]
    signals = proxy[["domain", "source"] + PROXY_SIGNALS].rename(
        columns={s: f"proxy:{s}" for s in PROXY_SIGNALS}
    )
    wide = wide.merge(signals, on=["domain", "source"])
    return wide


def znorm_within_domain(frame: pd.DataFrame, column: str) -> pd.Series:
    def z(group: pd.Series) -> pd.Series:
        std = group.std(ddof=1)
        return (group - group.mean()) / std if std > 0 else group * 0.0

    return frame.groupby("domain")[column].transform(z)


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(a, b)
    return float(rho), float(p)


def topk(a: Dict[str, float], b: Dict[str, float], k: int) -> float:
    top_a = set(sorted(a, key=a.get, reverse=True)[:k])
    top_b = set(sorted(b, key=b.get, reverse=True)[:k])
    return len(top_a & top_b) / k


def bootstrap_gain_ci(detail_path: Path, source: str, n_boot: int = 1000, seed: int = 17) -> Optional[List[float]]:
    """CI for G_i = pooled-NLL(full corpus minus i) - pooled-NLL(full) by
    resampling validation documents (same resample applied to both runs)."""
    if not detail_path.exists():
        return None
    detail = json.loads(detail_path.read_text())
    full = detail["runs"]["full"]
    ablated = detail["runs"].get(f"loso_{source}")
    if ablated is None:
        return None
    docs = sorted(full["per_doc"].keys())
    rng = np.random.default_rng(seed)
    deltas = []
    f_docs = np.array([full["per_doc"][d] for d in docs])
    a_docs = np.array([ablated["per_doc"][d] for d in docs])
    toks = np.array([full["per_doc_tokens"][d] for d in docs], dtype=float)
    for _ in range(n_boot):
        idx = rng.integers(0, len(docs), len(docs))
        f_sum = (f_docs[idx] * toks[idx]).sum()
        a_sum = (a_docs[idx] * toks[idx]).sum()
        deltas.append(a_sum / toks[idx].sum() - f_sum / toks[idx].sum())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return [float(lo), float(hi)]


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    tables = load_tables()
    wide = gains_wide(tables)

    methods = {
        "proxy_gain (paper L0)": "proxy:proxy_gain",
        "DQS": "proxy:dqs",
        "influence": "proxy:influence",
        "shapley": "proxy:shapley",
        "unified (fixed)": "proxy:unified",
        "row count": "proxy:row_count",
        "token count": "proxy:token_count_ws",
        "pythia-70m": "pythia-70m",
        "pythia-160m": "pythia-160m",
    }

    # ---------- 1. per-domain ranking agreement vs 410m ----------
    rows = []
    for label, column in methods.items():
        per_domain = {}
        for domain, group in wide.groupby("domain"):
            rho, p = spearman(group[column], group[REFERENCE])
            top1 = topk(
                dict(zip(group["source"], group[column])),
                dict(zip(group["source"], group[REFERENCE])),
                1,
            )
            per_domain[domain] = {"spearman": rho, "p": p, "top1": top1}
        mean_rho = float(np.nanmean([v["spearman"] for v in per_domain.values()]))
        mean_top1 = float(np.mean([v["top1"] for v in per_domain.values()]))
        rows.append(
            {
                "method": label,
                **{f"rho_{d}": round(per_domain[d]["spearman"], 3) for d in per_domain},
                "mean_rho": round(mean_rho, 3),
                "mean_top1": round(mean_top1, 3),
            }
        )
    ranking = pd.DataFrame(rows).sort_values("mean_rho", ascending=False)
    ranking.to_csv(RESULTS / "ranking_vs_410m.csv", index=False)

    # ---------- 2. pooled cross-scale correlation matrix ----------
    cols = {
        "proxy gain (L0)": "proxy:proxy_gain",
        "DQS": "proxy:dqs",
        "row count": "proxy:row_count",
        "token count": "proxy:token_count_ws",
        "pythia-70m": "pythia-70m",
        "pythia-160m": "pythia-160m",
        "pythia-410m": "pythia-410m",
    }
    z = pd.DataFrame({label: znorm_within_domain(wide, col) for label, col in cols.items()})
    matrix = z.corr(method="spearman")
    matrix.to_csv(RESULTS / "cross_scale_matrix.csv")
    pooled_p = {}
    for label, column in cols.items():
        if column == "pythia-410m":
            continue
        rho, p = spearman(z[label], z["pythia-410m"])
        pooled_p[label] = {"rho": round(rho, 3), "p": float(p)}

    # ---------- 3. gain magnitudes with bootstrap CIs ----------
    mag_rows = []
    for scale in LM_SCALES:
        for domain in wide["domain"].unique():
            sub = wide[wide["domain"] == domain]
            detail_path = RESULTS / f"lm_detail_{scale}_{domain}.json"
            for _, row in sub.iterrows():
                ci = bootstrap_gain_ci(detail_path, row["source"])
                mag_rows.append(
                    {
                        "scale": scale,
                        "params": LM_PARAMS[scale],
                        "domain": domain,
                        "source": row["source"],
                        "gain": row[scale],
                        "ci_low": ci[0] if ci else np.nan,
                        "ci_high": ci[1] if ci else np.nan,
                    }
                )
    magnitudes = pd.DataFrame(mag_rows)
    magnitudes.to_csv(RESULTS / "gain_magnitudes.csv", index=False)

    # ---------- 4. alpha calibration ----------
    alpha_rows = []
    n_70 = LM_PARAMS["pythia-70m"]
    n_410 = LM_PARAMS["pythia-410m"]
    for _, row in wide.iterrows():
        g70, g160, g410 = row["pythia-70m"], row["pythia-160m"], row["pythia-410m"]
        same_sign_410 = np.sign(g70) == np.sign(g410) and g70 != 0 and g410 != 0
        alpha_pair = np.log(g410 / g70) / np.log(n_70 / n_410) if same_sign_410 else np.nan
        same_sign_all = np.sign(g70) == np.sign(g160) == np.sign(g410) and min(abs(g70), abs(g160), abs(g410)) > 0
        if same_sign_all:
            slope = stats.linregress(
                np.log([n_70, LM_PARAMS["pythia-160m"], n_410]),
                np.log([abs(g70), abs(g160), abs(g410)]),
            ).slope
        else:
            slope = np.nan
        alpha_rows.append(
            {
                "domain": row["domain"],
                "source": row["source"],
                "gain_70m": g70,
                "gain_160m": g160,
                "gain_410m": g410,
                "alpha_pairwise": alpha_pair,
                "alpha_ols_3scale": slope,
                "sign_stable": bool(same_sign_all),
            }
        )
    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df.to_csv(RESULTS / "alpha_calibration.csv", index=False)

    # ---------- 5. top sources ----------
    top_rows = []
    for domain, group in wide.groupby("domain"):
        record = {"domain": domain}
        for label, column in {**methods, "pythia-410m": "pythia-410m"}.items():
            values = dict(zip(group["source"], group[column]))
            record[label] = max(values, key=values.get)
        top_rows.append(record)
    tops = pd.DataFrame(top_rows)
    tops.to_csv(RESULTS / "top_sources.csv", index=False)

    # ---------- 6. sign stability ----------
    sign_stable = {
        "70m_vs_410m": float((np.sign(wide["pythia-70m"]) == np.sign(wide["pythia-410m"])).mean()),
        "160m_vs_410m": float((np.sign(wide["pythia-160m"]) == np.sign(wide["pythia-410m"])).mean()),
        "proxy_vs_410m": float((np.sign(wide["proxy:proxy_gain"]) == np.sign(wide["pythia-410m"])).mean()),
    }

    # ---------- 7. figures ----------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    # Fig 1: ranking agreement bars
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    plot_rows = ranking.set_index("method")["mean_rho"]
    colors = ["#2563eb" if "pythia" in m else "#9ca3af" for m in plot_rows.index]
    ax.barh(plot_rows.index, plot_rows.values, color=colors)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(f"Mean Spearman $\\rho$ with {REFERENCE} LOSO gains (per-domain, n=6 each)")
    ax.set_xlim(-1, 1)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_ranking_agreement.pdf")
    fig.savefig(FIGURES / "fig1_ranking_agreement.png", dpi=200)
    plt.close(fig)

    # Fig 2: pooled scatter 70m vs 410m and proxy vs 410m
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    domain_colors = {"general_instruction": "#2563eb", "math_reasoning": "#dc2626", "code_summarization": "#059669"}
    for ax, wide_col, z_label, title in [
        (axes[0], "pythia-70m", "pythia-70m", "pythia-70m vs 410m"),
        (axes[1], "proxy:proxy_gain", "proxy gain (L0)", "proxy gain (L0) vs 410m"),
    ]:
        for domain, group in wide.groupby("domain"):
            ax.scatter(
                znorm_within_domain(group, wide_col),
                znorm_within_domain(group, REFERENCE),
                s=18,
                alpha=0.85,
                color=domain_colors[domain],
                label=domain.replace("_", " "),
            )
        rho, p = spearman(z[z_label], z["pythia-410m"])
        ax.set_title(f"{title}\npooled $\\rho$={rho:.2f} (p={p:.1e})", fontsize=8)
        ax.axhline(0, color="k", lw=0.4)
        ax.axvline(0, color="k", lw=0.4)
        ax.set_xlabel("within-domain z-score")
    axes[0].set_ylabel(f"z({REFERENCE} gain)")
    axes[1].legend(fontsize=6, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_pooled_scatter.pdf")
    fig.savefig(FIGURES / "fig2_pooled_scatter.png", dpi=200)
    plt.close(fig)

    # Fig 3: gain magnitude vs params (log-log)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for domain, group in magnitudes.groupby("domain"):
        med = group.groupby("params")["gain"].apply(lambda s: np.median(np.abs(s)))
        ax.plot(med.index, med.values, "o-", label=domain.replace("_", " "), color=domain_colors[domain])
    ratio = np.logspace(np.log10(7e7), np.log10(4.1e8), 10)
    anchor = np.median(np.abs(magnitudes[magnitudes["scale"] == "pythia-70m"]["gain"]))
    ax.plot(ratio, anchor * (ratio / n_70) ** -0.28, "k--", lw=0.8, label="power law $\\alpha$=0.28 (paper)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel("median |LOSO gain| (nats/token)")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_gain_scaling.pdf")
    fig.savefig(FIGURES / "fig3_gain_scaling.png", dpi=200)
    plt.close(fig)

    # ---------- 8. summary ----------
    alpha_pair_median = float(alpha_df["alpha_pairwise"].median())
    alpha_ols_median = float(alpha_df["alpha_ols_3scale"].median())
    shrink = float(
        np.median(np.abs(wide["pythia-410m"])) / np.median(np.abs(wide["pythia-70m"]))
    )
    summary = {
        "pooled_spearman_vs_410m": pooled_p,
        "per_domain_mean_rho": {r["method"]: r["mean_rho"] for _, r in ranking.iterrows()},
        "sign_stability": sign_stable,
        "alpha_pairwise_median": alpha_pair_median,
        "alpha_ols_median": alpha_ols_median,
        "gain_shrink_410m_vs_70m_median_ratio": shrink,
        "top_sources": tops.to_dict(orient="records"),
        "n_sources": int(len(wide)),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
