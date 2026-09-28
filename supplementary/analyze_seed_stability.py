"""Seed-stability analysis for the cross-scale supplementary experiment.

Compares LOSO gains from the primary run (seed 17, results/) with the
replication run (seed 43, results_seed43/) to separate genuine cross-scale
disagreement from ordinary fine-tuning noise:

  1. per-scale rank stability across seeds (per-domain and pooled Spearman)
  2. cross-scale agreement recomputed on seed-averaged gains

Outputs: results/seed_stability.csv, results/summary_seed.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS_SEED43 = HERE / "results_seed43"

LM_SCALES = ["pythia-70m", "pythia-160m", "pythia-410m"]
REFERENCE = "pythia-410m"


def load(seed_dir: Path) -> pd.DataFrame:
    frames = [
        pd.read_csv(seed_dir / f"lm_gains_{scale}.csv")[["domain", "source", "gain"]].rename(
            columns={"gain": scale}
        )
        for scale in LM_SCALES
    ]
    wide = frames[0]
    for frame in frames[1:]:
        wide = wide.merge(frame, on=["domain", "source"])
    return wide


def pooled_spearman(frame: pd.DataFrame, a: str, b: str) -> Dict[str, float]:
    def z(column: str) -> pd.Series:
        def zgroup(group: pd.Series) -> pd.Series:
            std = group.std(ddof=1)
            return (group - group.mean()) / std if std > 0 else group * 0.0

        return frame.groupby("domain")[column].transform(zgroup)

    rho, p = stats.spearmanr(z(a), z(b))
    return {"rho": round(float(rho), 3), "p": float(p)}


def main() -> None:
    primary = load(RESULTS).rename(columns={s: f"{s}_s17" for s in LM_SCALES})
    replica = load(RESULTS_SEED43).rename(columns={s: f"{s}_s43" for s in LM_SCALES})
    merged = primary.merge(replica, on=["domain", "source"])
    for scale in LM_SCALES:
        merged[scale] = (merged[f"{scale}_s17"] + merged[f"{scale}_s43"]) / 2

    rows = []
    for scale in LM_SCALES:
        per_domain = []
        for domain, group in merged.groupby("domain"):
            if group[f"{scale}_s17"].std() == 0 or group[f"{scale}_s43"].std() == 0:
                per_domain.append(np.nan)
                continue
            rho, _ = stats.spearmanr(group[f"{scale}_s17"], group[f"{scale}_s43"])
            per_domain.append(rho)
        pooled = pooled_spearman(merged, f"{scale}_s17", f"{scale}_s43")
        rows.append(
            {
                "scale": scale,
                "seed17_vs_seed43_per_domain_rho": [round(float(r), 3) if np.isfinite(r) else None for r in per_domain],
                "seed17_vs_seed43_mean_rho": round(float(np.nanmean(per_domain)), 3),
                "seed17_vs_seed43_pooled_rho": pooled["rho"],
                "seed17_vs_seed43_pooled_p": pooled["p"],
                "sign_agreement": float((np.sign(merged[f"{scale}_s17"]) == np.sign(merged[f"{scale}_s43"])).mean()),
            }
        )
    stability = pd.DataFrame(rows)
    stability.to_csv(RESULTS / "seed_stability.csv", index=False)

    # cross-scale agreement on seed-averaged gains
    proxy = pd.read_csv(RESULTS / "proxy_baseline.csv")[["domain", "source", "proxy_gain"]]
    merged = merged.merge(proxy, on=["domain", "source"])
    cross = {
        "proxy_gain_vs_410m_seedavg": pooled_spearman(merged, "proxy_gain", REFERENCE),
        "pythia70m_vs_410m_seedavg": pooled_spearman(merged, "pythia-70m", REFERENCE),
        "pythia160m_vs_410m_seedavg": pooled_spearman(merged, "pythia-160m", REFERENCE),
        "pythia70m_vs_160m_seedavg": pooled_spearman(merged, "pythia-70m", "pythia-160m"),
    }

    summary = {
        "seed_stability": rows,
        "cross_scale_seed_averaged": cross,
        "n_sources": int(len(merged)),
    }
    (RESULTS / "summary_seed.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # Figure: seed-stability vs cross-scale pooled rho, one bar group per scale
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    labels = ["pythia-70m", "pythia-160m", "pythia-410m"]
    seed_rho = [pooled_spearman(merged, f"{s}_s17", f"{s}_s43")["rho"] for s in labels]
    cross_map = {
        "pythia-70m": cross["pythia70m_vs_410m_seedavg"]["rho"],
        "pythia-160m": cross["pythia160m_vs_410m_seedavg"]["rho"],
        "pythia-410m": cross["proxy_gain_vs_410m_seedavg"]["rho"],
    }
    cross_rho = [cross_map[s] for s in labels]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, seed_rho, width=0.4, color="#9ca3af", label="same scale, seed 17 vs 43 (noise floor)")
    ax.bar(x + 0.2, cross_rho, width=0.4, color="#2563eb", label="vs seed-averaged 410m ranking")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["pythia-70m", "pythia-160m", "pythia-410m\n(bar = proxy L0 gain)"])
    ax.set_ylabel("pooled Spearman $\\rho$ (n=18)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "fig4_seed_stability.pdf")
    fig.savefig(HERE / "figures" / "fig4_seed_stability.png", dpi=200)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
