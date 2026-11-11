#!/usr/bin/env python3
"""Compact teaser version of the Pareto figure for use as a wrapfigure
in the intro. Smaller dimensions, smaller fonts, no in-plot legend
(legend moves to the caption). Same data as make_pareto_figure.py."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "current_paper" / "figures" / "pareto_h3_compact.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

DATA = {
    "dyMEAN":           {"aar": 0.431, "ddg": 0.524, "para": "full",  "is_PRiME": False},
    "dyAb":             {"aar": 0.335, "ddg": 0.511, "para": "full",  "is_PRiME": False},
    "AbEgDiffuser":     {"aar": 0.322, "ddg": 0.433, "para": "graft", "is_PRiME": False},
    "FlowDesign":       {"aar": 0.325, "ddg": 0.680, "para": "graft", "is_PRiME": False},
    "AbX":              {"aar": 0.279, "ddg": 0.610, "para": "graft", "is_PRiME": False},
    "PRiME":             {"aar": 0.531, "ddg": 0.669, "para": "full",  "is_PRiME": True},
    "PRiME-star":        {"aar": 0.727, "ddg": 0.586, "para": "full",  "is_PRiME": True},
}
LABEL = {
    "dyMEAN":       "dyMEAN",
    "dyAb":         r"dyAb $\dagger$",
    "AbEgDiffuser": r"AbEgDiffuser $\dagger\ddagger\star$",
    "FlowDesign":   r"FlowDesign $\dagger\ddagger$",
    "AbX":          r"AbX $\dagger\ddagger\star$",
    "PRiME":         r"$\mathit{PRiME}$",
    "PRiME-star":    r"$\mathit{PRiME}\,\star$",
}
points = [(m, d["aar"]*100, d["ddg"]*100,
           d["para"], d["is_PRiME"]) for m, d in DATA.items()]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.2, 2.6))

# Pareto frontier
pts_sorted = sorted(points, key=lambda p: -p[1])
pareto, max_y = [], -np.inf
for m, x, y, *_ in pts_sorted:
    if y > max_y:
        pareto.append((x, y))
        max_y = y
pareto.sort()
pareto_x = [p[0] for p in pareto]
pareto_y = [p[1] for p in pareto]

xmin, ymin = 22, 38
xmax = 82
ax.fill_between([xmin] + pareto_x + [xmax], ymin,
                [ymin] + pareto_y + [pareto_y[-1]],
                step="post", color="#f4f4f4", alpha=0.9, zorder=0)
ax.step([xmin] + pareto_x + [xmax], [ymin] + pareto_y + [pareto_y[-1]],
        where="post", color="#bbbbbb", linewidth=0.6,
        linestyle=(0, (3, 2)), zorder=1)

for method, x, y, para, is_PRiME in points:
    color = "#d62728" if is_PRiME else "#444444"
    marker = "o" if para == "full" else "^"
    edge = "black" if is_PRiME else color
    size = 75 if method == "PRiME-star" else (55 if method == "PRiME" else 38)
    lw = 1.2 if is_PRiME else 0.6
    ax.scatter(x, y, marker=marker, s=size,
               facecolor=color, edgecolor=edge, linewidth=lw, zorder=3)
    dx, dy, ha, va = 1.2, 0.4, "left", "bottom"
    if method == "FlowDesign":   dx, dy, ha, va =  0.0,  1.2, "center", "bottom"
    if method == "AbX":          dx, dy, ha, va =  1.4, -0.2, "left", "center"
    if method == "AbEgDiffuser": dx, dy, ha, va =  1.4, -0.2, "left", "center"
    if method == "dyMEAN":       dx, dy, ha, va =  1.4, -0.2, "left", "center"
    if method == "dyAb":         dx, dy, ha, va =  1.4, -0.2, "left", "center"
    if method == "PRiME-star":    dx, dy, ha, va = -1.4, -1.6, "right", "top"
    if method == "PRiME":         dx, dy, ha, va =  1.6, 0.4, "left", "bottom"
    ax.annotate(LABEL[method], (x + dx, y + dy),
                ha=ha, va=va,
                fontsize=7.5,
                fontweight="bold" if is_PRiME else "normal",
                fontstyle="italic" if is_PRiME else "normal",
                color=color, zorder=4)

ax.set_xlabel("AAR (%)", fontsize=8.5)
ax.set_ylabel(r"$\Delta\Delta G\!<\!0$ (%)", fontsize=8.5)
ax.set_xlim(xmin, xmax)
ax.set_ylim(38, 75)
ax.tick_params(axis="both", labelsize=7.5, length=3)
ax.grid(True, linestyle=":", linewidth=0.4, color="#dddddd", zorder=1)

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight", pad_inches=0.05)
plt.savefig(OUT.with_suffix(".png"), bbox_inches="tight", pad_inches=0.05, dpi=220)
print(f"wrote {OUT} and {OUT.with_suffix('.png')}")
