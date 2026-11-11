#!/usr/bin/env python3
"""Generate diagnostic figures for the three PRiME-star runs:

  1) pair_correlation.pdf   (3-panel Spearman heatmap, one per setting)
  2) aar_vs_structure.pdf   (3-panel AAR vs RMSD scatter, color = DockQ)
  3) per_cdr_breakdown.pdf  (boxplot, multi-CDR, with PLM-free reference)
  4) target_ranking_change.pdf (slope chart of per-target rank shift)
  5) time_vs_aar.pdf        (AAR vs deposition year, time-split only)

All inputs come from existing per-target / per-sample CSVs and the
SAbDab summary; no model retraining required.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "current_paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Data paths
# --------------------------------------------------------------------------
PATHS = {
    "H3": {
        "per_target": ROOT / "checkpoints/esm2_3B_eqadapter_rflow_adaptive_time/version_13/checkpoint/results_best/metrics/per_target_metrics.csv",
        "per_sample": ROOT / "checkpoints/esm2_3B_eqadapter_rflow_adaptive_time/version_13/checkpoint/results_best/metrics/per_sample_metrics.csv",
        "label": "RAbD CDR-H3",
    },
    "Multi": {
        "per_target": ROOT / "checkpoints/esm2_3B_eqadapter_rflow_adaptive_time_multi/version_1/checkpoint/results_best/metrics/per_target_metrics.csv",
        "per_sample": ROOT / "checkpoints/esm2_3B_eqadapter_rflow_adaptive_time_multi/version_1/checkpoint/results_best/metrics/per_sample_metrics.csv",
        "label": "Multi-CDR",
    },
    "Time": {
        "per_target": ROOT / "checkpoints/esm2_3B_eqadapter_rflow_adaptive_time_time/version_2/checkpoint/results_best_maskH3/metrics/per_target_metrics.csv",
        "per_sample": ROOT / "checkpoints/esm2_3B_eqadapter_rflow_adaptive_time_time/version_2/checkpoint/results_best_maskH3/metrics/per_sample_metrics.csv",
        "label": "Time-split CDR-H3",
    },
}
SABDAB = ROOT / "summaries/sabdab_summary.tsv"
PLM_FREE_MULTI = ROOT / "all_models_comparison_rabd_multicdrs.csv"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# --------------------------------------------------------------------------
# Figure 1: pair-correlation heatmap (Spearman) per setting
# --------------------------------------------------------------------------
def fig1_pair_correlation():
    metrics = [
        ("AAR H3_mean",          "AAR"),
        ("CAAR H3_mean",         "CAAR"),
        ("RMSD(CA) CDRH3 aligned_mean", "RMSD$_{H3}$"),
        ("RMSD(CA) aligned_mean", "RMSD$_{glob}$"),
        ("TMscore_mean",         "TM"),
        ("LDDT_mean",            "lDDT"),
        ("DockQ_mean",           "DockQ"),
        ("interface_clashes_mean", "Clashes"),
    ]
    # For Multi: AAR over all CDRs, RMSD over all CDRs (averaged H1..L3)
    multi_metric_overrides = {
        "AAR H3_mean": "AAR_mean",
        "CAAR H3_mean": "CAAR_mean",
        "RMSD(CA) CDRH3 aligned_mean": None,  # we'll compute mean of all-CDR
    }

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    for ax, (key, info) in zip(axes, PATHS.items()):
        df = pd.read_csv(info["per_target"])
        cols, labels = [], []
        for col, lab in metrics:
            if key == "Multi" and col in multi_metric_overrides:
                col2 = multi_metric_overrides[col]
                if col2 is None:
                    # Mean of aligned per-CDR RMSD
                    parts = ["RMSD(CA) CDRH1 aligned_mean", "RMSD(CA) CDRH2 aligned_mean",
                             "RMSD(CA) CDRH3 aligned_mean", "RMSD(CA) CDRL1 aligned_mean",
                             "RMSD(CA) CDRL2 aligned_mean", "RMSD(CA) CDRL3 aligned_mean"]
                    parts = [p for p in parts if p in df.columns]
                    if not parts:
                        continue
                    series = df[parts].mean(axis=1)
                    cols.append(series.values)
                    labels.append("RMSD$_{CDRs}$")
                    continue
                col = col2
            if col not in df.columns:
                continue
            cols.append(df[col].values)
            labels.append(lab)
        M = np.array(cols)
        n = M.shape[0]
        rho = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    rho[i, j] = 1.0
                else:
                    r, _ = spearmanr(M[i], M[j], nan_policy="omit")
                    rho[i, j] = r if np.isfinite(r) else 0.0
        im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_title(info["label"], fontsize=10)
        # annotate cells
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                val = rho[i, j]
                txtcolor = "white" if abs(val) > 0.55 else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=6.0, color=txtcolor)
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(False)
        ax.tick_params(which="minor", length=0)
        # frame
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(True)

    cbar = fig.colorbar(im, ax=axes, orientation="vertical", shrink=0.85,
                        fraction=0.025, pad=0.02)
    cbar.set_label(r"Spearman $\rho$", fontsize=9)
    cbar.ax.tick_params(labelsize=7.5)

    out = FIGDIR / "diagnostic_pair_correlation.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"wrote {out.name}")

# --------------------------------------------------------------------------
# Figure 2: per-target AAR vs RMSD_H3, color = DockQ (3 panels)
# --------------------------------------------------------------------------
def fig2_aar_vs_structure():
    # Drop time-split: its per-target file uses a larger eval set whose
    # absolute AAR does not match the paper's curated time-split summary.
    keys = [k for k in PATHS if k != "Time"]
    fig, axes = plt.subplots(1, len(keys), figsize=(7.6, 3.4), sharey=False)

    for ax, key in zip(axes, keys):
        info = PATHS[key]
        df = pd.read_csv(info["per_target"])
        if key == "Multi":
            x_col, y_col = "AAR_mean", "RMSD(CA) CDRH3 aligned_mean"
        else:
            x_col, y_col = "AAR H3_mean", "RMSD(CA) CDRH3 aligned_mean"
        if x_col not in df.columns or y_col not in df.columns:
            ax.set_visible(False)
            continue
        x = df[x_col].values * 100
        y = df[y_col].values
        c = df["DockQ_mean"].values if "DockQ_mean" in df.columns else None

        sc = ax.scatter(x, y, c=c, cmap="viridis", vmin=0, vmax=1,
                        s=32, edgecolor="white", linewidth=0.5, alpha=0.95)

        # mean lines
        ax.axhline(np.nanmean(y), color="#888", linewidth=0.5, linestyle=":")
        ax.axvline(np.nanmean(x), color="#888", linewidth=0.5, linestyle=":")
        ax.set_xlabel("AAR (%)" if key != "Multi" else "AAR all-CDR (%)")
        ax.set_ylabel(r"RMSD$_{H3}$ (\AA)")
        ax.set_title(info["label"], fontsize=10)
        ax.grid(True, linestyle=":", linewidth=0.4, color="#dddddd")

    cbar = fig.colorbar(sc, ax=axes, orientation="vertical", shrink=0.85,
                        fraction=0.020, pad=0.02)
    cbar.set_label("DockQ", fontsize=9)
    cbar.ax.tick_params(labelsize=7.5)

    out = FIGDIR / "diagnostic_aar_vs_structure.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"wrote {out.name}")

# --------------------------------------------------------------------------
# Figure 3: per-CDR AAR boxplot (multi only) + PLM-free reference line
# --------------------------------------------------------------------------
def fig3_per_cdr_breakdown():
    df = pd.read_csv(PATHS["Multi"]["per_target"])
    cdr_cols = [
        ("AAR H1_mean", "H1"),
        ("AAR H2_mean", "H2"),
        ("AAR H3_mean", "H3"),
        ("AAR L1_mean", "L1"),
        ("AAR L2_mean", "L2"),
        ("AAR L3_mean", "L3"),
    ]
    data = [df[c].values * 100 for c, _ in cdr_cols if c in df.columns]
    labels = [lab for c, lab in cdr_cols if c in df.columns]

    # PLM-free baseline values from all_models_comparison_rabd_multicdrs.csv
    plm_free = {}
    if PLM_FREE_MULTI.exists():
        cmp = pd.read_csv(PLM_FREE_MULTI)
        for cdr in labels:
            row = cmp[cmp["Metric"] == f"AAR {cdr}"]
            if len(row) > 0 and "no_adapter_mean" in row.columns:
                v = row["no_adapter_mean"].iloc[0]
                if pd.notna(v):
                    plm_free[cdr] = float(v) * 100

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bp = ax.boxplot(data, labels=labels, widths=0.55, patch_artist=True,
                    medianprops=dict(color="#222", linewidth=1.4),
                    whiskerprops=dict(color="#666", linewidth=0.7),
                    capprops=dict(color="#666", linewidth=0.7),
                    flierprops=dict(marker="o", markersize=3,
                                    markerfacecolor="#bbb",
                                    markeredgecolor="none"))
    for patch in bp["boxes"]:
        patch.set_facecolor("#fce7e6")
        patch.set_edgecolor("#d62728")
        patch.set_linewidth(1.1)

    # Overlay individual targets as faint dots (jitter)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        jitter = rng.uniform(-0.13, 0.13, size=len(vals))
        ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals,
                   s=10, color="#444", alpha=0.45, zorder=2)

    # PLM-free single-value reference (mean only -- per-target data not
    # available for the multi-CDR PLM-free run, so we render it as a
    # distinct marker rather than a line that looks like a distribution).
    if plm_free:
        for i, lab in enumerate(labels, start=1):
            if lab in plm_free:
                ax.scatter([i], [plm_free[lab]], marker="D", s=55,
                           facecolor="#ffffff", edgecolor="#1f77b4",
                           linewidth=1.4, zorder=4)
        # PLM lift arrows: from PLM-free mean to PRiME-star median
        for i, (lab, vals) in enumerate(zip(labels, data), start=1):
            if lab in plm_free:
                lift_lo = plm_free[lab]
                lift_hi = float(np.median(vals))
                if abs(lift_hi - lift_lo) > 0.5:
                    ax.annotate(
                        "", xy=(i + 0.34, lift_hi),
                        xytext=(i + 0.34, lift_lo),
                        arrowprops=dict(
                            arrowstyle="-|>", color="#1f77b4",
                            lw=1.0, alpha=0.7,
                            shrinkA=0, shrinkB=0,
                        ),
                        zorder=3,
                    )
                    ax.text(i + 0.40,
                            (lift_hi + lift_lo) / 2,
                            f"+{lift_hi - lift_lo:.1f}",
                            fontsize=6.5, color="#1f77b4",
                            ha="left", va="center", zorder=4)
        # legend handles
        h_ours = mpatches.Patch(facecolor="#fce7e6", edgecolor="#d62728",
                                label=r"$\mathit{PRiME}^\star$   (box, dots)")
        h_plm = plt.Line2D([0], [0], marker="D", color="#1f77b4",
                           linestyle="None", markerfacecolor="#ffffff",
                           markeredgecolor="#1f77b4", markeredgewidth=1.4,
                           markersize=6,
                           label=r"$\mathit{PRiME}$ ")
        h_lift = plt.Line2D([0], [0], color="#1f77b4", linewidth=1.0,
                            alpha=0.7, label="median lift (pp)")
        ax.legend(handles=[h_ours, h_plm, h_lift], loc="lower left",
                  fontsize=7.5, frameon=True, framealpha=0.95,
                  edgecolor="#cccccc")

    ax.set_xlabel("CDR loop")
    ax.set_ylabel("AAR (%)")
    ax.set_ylim(20, 108)
    ax.set_xlim(0.4, len(labels) + 0.95)
    ax.set_title("Per-CDR AAR (multi-CDR, RAbD): "
                 r"$\mathit{PRiME}^\star$  vs $\mathit{PRiME}$ ",
                 fontsize=9.5)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.4, color="#dddddd")

    plt.tight_layout()
    out = FIGDIR / "diagnostic_per_cdr.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"wrote {out.name}")

# --------------------------------------------------------------------------
# Figure 4: per-target ranking change H3 -> Multi -> Time
# --------------------------------------------------------------------------
def fig4_target_ranking_change():
    # Time-split test set has different PDBs (chronological), so it does
    # not share targets with RAbD: we restrict to H3 -> Multi.
    h3 = pd.read_csv(PATHS["H3"]["per_target"])[["pdb", "AAR H3_mean"]]
    multi = pd.read_csv(PATHS["Multi"]["per_target"])[["pdb", "AAR_mean", "AAR H3_mean"]]
    h3 = h3.rename(columns={"AAR H3_mean": "H3"})
    multi_h3 = multi[["pdb", "AAR H3_mean"]].rename(columns={"AAR H3_mean": "Multi-H3"})
    multi_all = multi[["pdb", "AAR_mean"]].rename(columns={"AAR_mean": "Multi-all"})
    df = h3.merge(multi_h3, on="pdb").merge(multi_all, on="pdb")
    cols = ["H3", "Multi-H3", "Multi-all"]

    # ranks: highest AAR -> rank 1
    rank_df = df.copy()
    for c in cols:
        rank_df[c + "_rank"] = rank_df[c].rank(ascending=False, method="min")

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    n = len(rank_df)
    xs = np.arange(len(cols))

    # Color by H3-only AAR (so the eye picks out high-AAR targets)
    h3_aar = rank_df["H3"].values
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=h3_aar.min(), vmax=h3_aar.max())

    for i, row in rank_df.iterrows():
        ys = [row[c + "_rank"] for c in cols]
        ax.plot(xs, ys, color=cmap(norm(row["H3"])), linewidth=0.9, alpha=0.7)
        ax.scatter(xs, ys, color=cmap(norm(row["H3"])), s=14, zorder=3,
                   edgecolor="white", linewidth=0.4)

    ax.set_xticks(xs)
    nice = {"H3": "H3-only\nAAR$_{H3}$",
            "Multi-H3": "Multi-CDR\nAAR$_{H3}$",
            "Multi-all": "Multi-CDR\nAAR$_{all}$"}
    ax.set_xticklabels([nice.get(c, c) for c in cols])
    ax.invert_yaxis()
    ax.set_ylabel(f"AAR rank (1 = best, of {n} RAbD targets)")
    ax.set_title("Per-target AAR rank shift: H3-only $\\to$ Multi-CDR",
                 fontsize=10)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.4, color="#dddddd")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.7,
                        fraction=0.04, pad=0.02)
    cbar.set_label("H3-only AAR", fontsize=9)
    cbar.ax.tick_params(labelsize=7.5)

    plt.tight_layout()
    out = FIGDIR / "diagnostic_ranking_change.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"wrote {out.name}")

# --------------------------------------------------------------------------
# Figure 5: AAR vs deposition year (time-split)
# --------------------------------------------------------------------------
def fig5_time_vs_aar():
    # Disabled: the time-split per-target file uses a larger evaluation
    # set whose absolute AAR does not match the paper's headline number.
    # Will revisit once the curated subset is available.
    print("skipping fig5: time-split eval-set scope mismatch")
    return
    if not SABDAB.exists():
        print("skipping fig5: sabdab summary not found")
        return
    df = pd.read_csv(PATHS["Time"]["per_target"])
    sab = pd.read_csv(SABDAB, sep="\t", low_memory=False)
    # Take earliest date per pdb (sometimes multi-row)
    sab["date"] = pd.to_datetime(sab["date"], format="%m/%d/%y", errors="coerce")
    yearmap = sab.groupby("pdb", as_index=False)["date"].min()
    yearmap["year"] = yearmap["date"].dt.year
    df = df.merge(yearmap[["pdb", "year"]], on="pdb", how="left")
    df = df.dropna(subset=["year", "AAR H3_mean"])
    if df.empty:
        print("skipping fig5: no year matches")
        return

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = df["year"].values
    y = df["AAR H3_mean"].values * 100

    ax.scatter(x, y, s=28, color="#d62728", alpha=0.85,
               edgecolor="white", linewidth=0.5, zorder=3)

    # Per-year mean and 95% bootstrap CI
    years = np.sort(np.unique(x))
    means, los, his = [], [], []
    rng = np.random.default_rng(0)
    for yr in years:
        vals = y[x == yr]
        means.append(vals.mean())
        if len(vals) >= 2:
            boot = rng.choice(vals, size=(2000, len(vals)), replace=True).mean(axis=1)
            los.append(np.percentile(boot, 2.5))
            his.append(np.percentile(boot, 97.5))
        else:
            los.append(vals.mean())
            his.append(vals.mean())
    ax.plot(years, means, color="#444", linewidth=1.4, zorder=2)
    ax.fill_between(years, los, his, color="#bbb", alpha=0.35, zorder=1)

    # Linear fit for headline
    if len(np.unique(x)) >= 3:
        slope, intercept = np.polyfit(x, y, 1)
        xline = np.array([x.min(), x.max()])
        yline = slope * xline + intercept
        ax.plot(xline, yline, color="#1f77b4", linewidth=1.2,
                linestyle="--", zorder=2,
                label=f"linear fit: {slope:+.2f} %/yr")
        ax.legend(loc="lower left", fontsize=8, frameon=True,
                  framealpha=0.95, edgecolor="#cccccc")

    ax.set_xlabel("PDB deposition year")
    ax.set_ylabel("AAR (%)")
    ax.set_title("Per-target AAR vs deposition year (time-split, PRiME$^\\star$)",
                 fontsize=10)
    ax.grid(True, linestyle=":", linewidth=0.4, color="#dddddd")
    ax.set_ylim(20, 100)

    plt.tight_layout()
    out = FIGDIR / "diagnostic_time_vs_aar.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"wrote {out.name}")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    fig1_pair_correlation()
    fig2_aar_vs_structure()
    fig3_per_cdr_breakdown()
    fig4_target_ranking_change()
    fig5_time_vs_aar()
    print("done.")
