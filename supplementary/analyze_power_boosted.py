"""Power-boosted confirmatory analysis (conditions C1-C5).

Pre-specified criteria (fixed before running; see SUPPLEMENTARY.md S3.5):
  C1 LM-proxy condition  : seed-averaged 70m/160m ranking predicts seed-averaged
                           410m ranking, pooled Spearman, permutation p < 0.05
  C2 paper-proxy condition: L0 logistic-regression proxy_gain predicts
                           seed-averaged 410m ranking, p < 0.05
  C3 screening condition : top-25% source identification above chance
                           (permutation p < 0.05) and sign AUC > 0.6
  C4 domain-conditional  : per-domain Spearman (n=8), exploratory
  C5 noise floor         : seed-to-seed stability of each scale's own ranking

Inputs : results_large/proxy_baseline.csv, results_large_seed*/lm_gains_*.csv
Outputs: results_large/power_conditions.csv, results_large/summary_power.json,
         figures/fig5_power_boosted.{pdf,png}
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_large"
FIGURES = HERE / "figures"

SEEDS = [17, 43, 7, 101, 2025]
LM_SCALES = ["pythia-70m", "pythia-160m", "pythia-410m"]
N_PERM = 10_000
RNG = np.random.default_rng(20260926)


def load_gains(out_dir: Path) -> pd.DataFrame:
    frames = [
        pd.read_csv(out_dir / f"lm_gains_{scale}.csv")[["domain", "source", "gain"]].rename(
            columns={"gain": scale}
        )
        for scale in LM_SCALES
    ]
    wide = frames[0]
    for frame in frames[1:]:
        wide = wide.merge(frame, on=["domain", "source"])
    return wide


def znorm(frame: pd.DataFrame, column: str) -> np.ndarray:
    def z(group: pd.Series) -> pd.Series:
        std = group.std(ddof=1)
        return (group - group.mean()) / std if std > 0 else group * 0.0

    return frame.groupby("domain")[column].transform(z).to_numpy()


def spearman(a, b) -> float:
    return float(stats.spearmanr(a, b).statistic)


def perm_p(frame: pd.DataFrame, pred_col: str, ref_col: str) -> float:
    """Within-domain permutation test for pooled Spearman (preserves domain strata)."""
    obs = spearman(znorm(frame, pred_col), znorm(frame, ref_col))
    domains = frame["domain"].to_numpy()
    pred = znorm(frame, pred_col)
    ref = znorm(frame, ref_col)
    count = 0
    for _ in range(N_PERM):
        shuffled = pred.copy()
        for domain in np.unique(domains):
            mask = domains == domain
            shuffled[mask] = RNG.permutation(shuffled[mask])
        if spearman(shuffled, ref) >= obs:
            count += 1
    return (count + 1) / (N_PERM + 1)


def bootstrap_rho_ci(frame: pd.DataFrame, pred_col: str, ref_col: str, n_boot: int = 2000) -> List[float]:
    """Stratified bootstrap over sources within each domain."""
    domains = sorted(frame["domain"].unique())
    rhos = []
    for _ in range(n_boot):
        parts = [frame[frame["domain"] == d].sample(frac=1.0, replace=True, random_state=int(RNG.integers(1e9))) for d in domains]
        sample = pd.concat(parts)
        rhos.append(spearman(znorm(sample, pred_col), znorm(sample, ref_col)))
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return [float(lo), float(hi)]


def topk_hits(frame: pd.DataFrame, pred_col: str, ref_col: str, k: int = 2) -> int:
    hits = 0
    for _, group in frame.groupby("domain"):
        ref_top = set(group.nlargest(k, ref_col)["source"])
        pred_top = set(group.nlargest(k, pred_col)["source"])
        hits += len(ref_top & pred_top)
    return hits


def topk_perm_p(frame: pd.DataFrame, pred_col: str, ref_col: str, k: int = 2) -> float:
    obs = topk_hits(frame, pred_col, ref_col, k)
    count = 0
    for _ in range(N_PERM):
        shuffled = frame.copy()
        for _, group_idx in frame.groupby("domain").groups.items():
            idx = list(group_idx)
            shuffled.loc[idx, pred_col] = shuffled.loc[idx, pred_col].to_numpy()[RNG.permutation(len(idx))]
        if topk_hits(shuffled, pred_col, ref_col, k) >= obs:
            count += 1
    return (count + 1) / (N_PERM + 1)


def sign_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    per_seed = {s: load_gains(HERE / f"results_large_seed{s}") for s in SEEDS}
    # seed-averaged gains
    avg = per_seed[SEEDS[0]][["domain", "source"]].copy()
    for scale in LM_SCALES:
        avg[scale] = sum(per_seed[s][scale] for s in SEEDS) / len(SEEDS)
    proxy = pd.read_csv(RESULTS / "proxy_baseline.csv")[["domain", "source", "proxy_gain", "dqs", "token_count_ws"]]
    frame = avg.merge(proxy, on=["domain", "source"])

    verdicts: List[Dict[str, object]] = []

    def record(cid, condition, rho, p, ci, verdict):
        verdicts.append(
            {"id": cid, "condition": condition, "rho_or_effect": rho, "perm_p": p, "ci95": ci, "verdict": verdict}
        )

    # ---------- C5: noise floor ----------
    c5 = {}
    for scale in LM_SCALES:
        pairwise = [
            spearman(znorm(per_seed[a], scale), znorm(per_seed[b], scale))
            for a, b in itertools.combinations(SEEDS, 2)
        ]
        c5[scale] = {"mean_pairwise_rho": round(float(np.mean(pairwise)), 3), "min": round(float(np.min(pairwise)), 3)}
    record("C5", "noise floor: 410m seed stability (mean pairwise pooled rho)", c5["pythia-410m"]["mean_pairwise_rho"], None, None, "interpretive")

    # ---------- C1: LM-proxy condition ----------
    for scale in ["pythia-70m", "pythia-160m"]:
        rho = spearman(znorm(frame, scale), znorm(frame, "pythia-410m"))
        p = perm_p(frame, scale, "pythia-410m")
        ci = bootstrap_rho_ci(frame, scale, "pythia-410m")
        record(f"C1-{scale}", f"seed-avg {scale} -> seed-avg 410m (pooled rho)", round(rho, 3), round(p, 4), [round(x, 3) for x in ci], "HOLDS" if p < 0.05 else "not established")

    # ---------- C2: paper's proxy ----------
    for col, label in [("proxy_gain", "L0 proxy gain"), ("dqs", "DQS"), ("token_count_ws", "token count")]:
        rho = spearman(znorm(frame, col), znorm(frame, "pythia-410m"))
        p = perm_p(frame, col, "pythia-410m")
        ci = bootstrap_rho_ci(frame, col, "pythia-410m")
        record(f"C2-{label}", f"{label} -> seed-avg 410m (pooled rho)", round(rho, 3), round(p, 4), [round(x, 3) for x in ci], "HOLDS" if p < 0.05 else "not established")

    # ---------- C3: screening ----------
    n_domains = frame["domain"].nunique()
    chance = 2 * 2 / 8 * n_domains  # expected top-2 overlap of two random 2-subsets of 8
    for col, label in [("proxy_gain", "L0 proxy"), ("pythia-70m", "pythia-70m"), ("pythia-160m", "pythia-160m")]:
        hits = topk_hits(frame, col, "pythia-410m", k=2)
        p = topk_perm_p(frame, col, "pythia-410m", k=2)
        auc = sign_auc(frame[col].to_numpy(), (frame["pythia-410m"].to_numpy() > 0).astype(int))
        record(
            f"C3-{label}",
            f"{label}: top-2/8 hits (chance={chance:.1f}) & sign AUC",
            f"hits={hits}/6, AUC={auc:.2f}",
            round(p, 4),
            None,
            "HOLDS" if (p < 0.05 and auc > 0.6) else "not established",
        )

    # ---------- C4: per-domain ----------
    c4 = {}
    for col in ["proxy_gain", "pythia-70m", "pythia-160m"]:
        c4[col] = {}
        for domain, group in frame.groupby("domain"):
            rho, p = stats.spearmanr(group[col], group["pythia-410m"])
            c4[col][domain] = {"rho": round(float(rho), 3), "p": round(float(p), 4)}

    table = pd.DataFrame(verdicts)
    table.to_csv(RESULTS / "power_conditions.csv", index=False)
    pd.DataFrame(c4).to_csv(RESULTS / "per_domain_power.csv")

    # magnitudes
    mags = {s: float(np.median(np.abs(frame[s]))) for s in LM_SCALES}

    summary = {
        "pre_registered_criteria": "pooled Spearman permutation p<0.05 (C1/C2); top-k perm p<0.05 and AUC>0.6 (C3)",
        "n_sources": int(len(frame)),
        "n_seeds": len(SEEDS),
        "verdicts": verdicts,
        "noise_floor_c5": c5,
        "per_domain_c4": c4,
        "median_abs_gain": mags,
    }
    (RESULTS / "summary_power.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---------- figure ----------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    ax = axes[0]
    bars = {
        "70m→410m": ("pythia-70m", "#93c5fd"),
        "160m→410m": ("pythia-160m", "#2563eb"),
        "proxy→410m": ("proxy_gain", "#f59e0b"),
        "DQS→410m": ("dqs", "#9ca3af"),
        "tok→410m": ("token_count_ws", "#d1d5db"),
    }
    names = list(bars.keys())
    rhos = [spearman(znorm(frame, bars[n][0]), znorm(frame, "pythia-410m")) for n in names]
    ps = [perm_p(frame, bars[n][0], "pythia-410m") for n in names]
    stars = ["**" if p < 0.01 else ("*" if p < 0.05 else "") for p in ps]
    colors = [bars[n][1] for n in names]
    ax.bar(names, rhos, color=colors)
    for i, (r, s) in enumerate(zip(rhos, stars)):
        ax.text(i, r + (0.045 if r >= 0 else -0.045), f"{r:.2f}{s}", ha="center", va="bottom" if r >= 0 else "top", fontsize=7.5)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(-0.35, 0.75)
    ax.set_ylabel("pooled Spearman $\\rho$ vs 410m (n=24)")
    ax.set_title("Power-boosted cross-scale agreement", fontsize=9)
    ax.tick_params(axis="x", labelrotation=0, labelsize=7.5)

    ax = axes[1]
    seed_names = LM_SCALES
    floor = [c5[s]["mean_pairwise_rho"] for s in seed_names]
    ax.bar(seed_names, floor, color=["#bbf7d0", "#86efac", "#16a34a"])
    for i, v in enumerate(floor):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean pairwise pooled $\\rho$ across 5 seeds")
    ax.set_title("Noise floor at 480-doc budget", fontsize=9)
    ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_power_boosted.pdf")
    fig.savefig(FIGURES / "fig5_power_boosted.png", dpi=200)
    print("fig5 written")


if __name__ == "__main__":
    main()
