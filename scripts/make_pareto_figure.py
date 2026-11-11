#!/usr/bin/env python3
"""Pareto scatter: AAR vs completion-corrected dG<0 on RAbD CDR-H3.

Values are hard-coded from the published tables in the paper
(tab:main_results for AAR, tab:proxy_metrics for ddG<0) so the figure
matches the prose exactly. Produces a clean single-panel PDF in
figures/pareto_h3.pdf.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "current_paper" / "figures" / "pareto_h3.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

# (AAR, ddG<0 corrected) for RAbD CDR-H3, both as fractions in [0,1].
# Sources:
#   AAR: tab:main_results, CDR-H3 column.
#   ddG<0: tab:proxy_metrics, CDR-H3 column (completion-corrected).
DATA = {
    "dyMEAN":           {"aar": 0.431, "ddg": 0.524, "para": "full",  "is_rime": False},
    "dyAb":             {"aar": 0.335, "ddg": 0.511, "para": "full",  "is_rime": False},
    "AbEgDiffuser":     {"aar": 0.322, "ddg": 0.433, "para": "graft", "is_rime": False},
    "FlowDesign":       {"aar": 0.325, "ddg": 0.680, "para": "graft", "is_rime": False},
    "AbX":              {"aar": 0.279, "ddg": 0.610, "para": "graft", "is_rime": False},
    "RiME":             {"aar": 0.531, "ddg": 0.669, "para": "full",  "is_rime": True},   # PLM-free
    "RiME-star":        {"aar": 0.727, "ddg": 0.586, "para": "full",  "is_rime": True},   # PLM-aug
}
LABEL = {
    "dyMEAN":       "dyMEAN",
    "dyAb":         r"dyAb $\dagger$",
    "AbEgDiffuser": r"AbEgDiffuser $\dagger\ddagger\star$",
    "FlowDesign":   r"FlowDesign $\dagger\ddagger$",
    "AbX":          r"AbX $\dagger\ddagger\star$",
    "RiME":         r"$\mathit{RiME}$",
    "RiME-star":    r"$\mathit{RiME}\,\star$",
}
points = [(m, d["aar"] * 100, d["ddg"] * 100,
           d["para"], d["is_rime"]) for m, d in DATA.items()]

# ----- Plot --------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(5.0, 4.0))

# Compute the Pareto frontier (upper-right preferred on both axes).
sorted_by_x = sorted(points, key=lambda p: -p[1])  # AAR desc
pareto, max_y = [], -np.inf
for m, x, y, *_ in sorted_by_x:
    if y > max_y:
        pareto.append((x, y))
        max_y = y
pareto.sort()  # by AAR asc
pareto_x = [p[0] for p in pareto]
pareto_y = [p[1] for p in pareto]

# Shade the Pareto-dominated region (lower-left of the staircase) faintly.
xmin, ymin = 22, 38
ax.fill_between([xmin] + pareto_x + [80], ymin,
                [ymin] + pareto_y + [pareto_y[-1]],
                step="post", color="#f4f4f4", alpha=0.9, zorder=0)
# Pareto frontier as a faint dashed line so the staircase is visible
ax.step([xmin] + pareto_x + [80], [ymin] + pareto_y + [pareto_y[-1]],
        where="post", color="#bbbbbb", linewidth=0.8,
        linestyle=(0, (4, 3)), zorder=1)

# Plot points
for method, x, y, para, is_rime in points:
    color = "#d62728" if is_rime else "#444444"
    marker = "o" if para == "full" else "^"
    edge = "black" if is_rime else color
    size = 150 if method == "RiME-star" else (110 if method == "RiME" else 85)
    lw = 1.6 if is_rime else 0.8
    ax.scatter(x, y, marker=marker, s=size,
               facecolor=color, edgecolor=edge, linewidth=lw, zorder=3)
    # Label offsets to avoid overlap
    dx, dy, ha, va = 1.6, 0.6, "left", "bottom"
    if method == "FlowDesign":   dx, dy, ha, va =  0.0,  1.4, "center", "bottom"
    if method == "AbX":          dx, dy, ha, va =  1.6, -0.2, "left", "center"
    if method == "AbEgDiffuser": dx, dy, ha, va =  0.0, -1.6, "center", "top"
    if method == "dyMEAN":       dx, dy, ha, va =  1.6, -0.2, "left", "center"
    if method == "dyAb":         dx, dy, ha, va =  1.6, -0.2, "left", "center"
    if method == "RiME-star":    dx, dy, ha, va = -1.6, -2.0, "right", "top"
    if method == "RiME":         dx, dy, ha, va =  1.8, 0.6, "left", "bottom"
    ax.annotate(LABEL[method], (x + dx, y + dy),
                ha=ha, va=va,
                fontsize=9.5,
                fontweight="bold" if is_rime else "normal",
                fontstyle="italic" if is_rime else "normal",
                color=color, zorder=4)

# Axes
ax.set_xlabel("AAR  (sequence recovery, %)", fontsize=11)
ax.set_ylabel(r"$\Delta\Delta G\!<\!0$  (completion-corrected, %)",
              fontsize=11)
ax.set_xlim(22, 80)
ax.set_ylim(38, 75)
ax.grid(True, linestyle=":", linewidth=0.5, color="#dddddd", zorder=1)

# Legend (paradigm + RiME family)
leg_elems = [
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#444444", markeredgecolor="#444444",
               markersize=8, label="full-structure"),
    plt.Line2D([0], [0], marker="^", color="w",
               markerfacecolor="#444444", markeredgecolor="#444444",
               markersize=8, label="CDR-grafting ($\\ddagger$)"),
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#d62728", markeredgecolor="black",
               markersize=10, markeredgewidth=1.4, label="RiME family (ours; $\\star$=PLM)"),
    mpatches.Patch(facecolor="#ffffff", edgecolor="none",
                   label="$\\dagger$ = extra inference-time conditioning"),
    mpatches.Patch(facecolor="#f4f4f4", edgecolor="none",
                   label="Pareto-dominated region"),
]
ax.legend(handles=leg_elems, loc="lower right", fontsize=8,
          frameon=True, framealpha=0.95, edgecolor="#cccccc",
          handletextpad=0.5)

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
plt.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=200)
print(f"wrote {OUT} and {OUT.with_suffix('.png')}")
