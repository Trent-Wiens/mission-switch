#!/usr/bin/env python3
"""Look at the whole grid at once, as pictures rather than rows.

    python3 heatmap.py                        defer, h=1, tau=0.3, margin
    python3 heatmap.py --metric jolt
    python3 heatmap.py --policy prepare --h 2 --tau 0.5
    python3 heatmap.py --all                  every policy x h x tau

Every row of grid_results.csv is one point in a 24 bearings x 18 ranges plane,
repeated per policy / h / tau / method.  Scrolling 46,656 rows tells you
nothing; the same data as four panels tells you where the method fails and at
what geometry.

Needs matplotlib.  Reads Images/grid_results.csv, writes PNG + PDF to Images/.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(os.path.dirname(HERE), "Images")
CSV_PATH = os.path.join(IMAGES, "grid_results.csv")

METHODS = ("cold", "warm", "blend", "filtered")
LABELS = {"cold": "Cold start", "warm": "Warm start",
          "blend": "Blend 0.5 s", "filtered": "Filtered 0.5 s"}

#: Diverging: one hue each side of a meaningful zero, neutral grey at the
#: midpoint.  Never a rainbow -- the midpoint has to read as "neither".
DIVERGING = LinearSegmentedColormap.from_list(
    "margin", ["#8F1F19", "#D0342C", "#CFCFCF", "#1F8A70", "#0E4A3C"])
#: Sequential: one hue, light to dark, for a quantity with no meaningful zero.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "jolt", ["#EEF1F6", "#8FB0E4", "#2E6FD9", "#17386E"])

METRICS = {
    "margin": {
        "column": "min_margin_after_commit",
        "label": "smallest safety margin after commit (m)",
        "diverging": True,
    },
    "jolt": {
        "column": "cmd_vel_jump",
        "label": "command jump at handover (m/s)",
        "diverging": False,
    },
    "turn": {
        "column": "turn_angle_at_commit",
        "label": "turn angle at commit (degrees)",
        "diverging": False,
    },
}


def load(policy, h, tau):
    rows = []
    with open(CSV_PATH) as fh:
        for r in csv.DictReader(fh):
            if r["policy"] != policy:
                continue
            if abs(float(r["h"]) - h) > 1e-9 or abs(float(r["tau"]) - tau) > 1e-9:
                continue
            rows.append(r)
    return rows


def panel(ax, rows, method, metric, vmin, vmax):
    sel = [r for r in rows if r["method"] == method]
    bearings = sorted({float(r["bearing_deg"]) for r in sel})
    ranges = sorted({float(r["detect_range"]) for r in sel})
    bi = {b: i for i, b in enumerate(bearings)}
    ri = {v: i for i, v in enumerate(ranges)}

    grid = [[float("nan")] * len(ranges) for _ in bearings]
    caught = []
    for r in sel:
        b, v = bi[float(r["bearing_deg"])], ri[float(r["detect_range"])]
        raw = r[metric["column"]]
        grid[b][v] = float("nan") if raw == "" else float(raw)
        if int(r["captured"]):
            caught.append((v, b))

    if metric["diverging"]:
        lim = max(abs(vmin), abs(vmax)) or 1.0
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        cmap = DIVERGING
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = SEQUENTIAL

    im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, norm=norm,
                   interpolation="nearest")

    # Captures marked by form as well as colour, so the picture does not rely
    # on hue alone.
    if caught:
        ax.scatter([c[0] for c in caught], [c[1] for c in caught],
                   marker="x", s=9, linewidths=0.7, color="#111111", alpha=0.75)

    ax.set_xticks(range(0, len(ranges), 3))
    ax.set_xticklabels([f"{ranges[i]:.0f}" for i in range(0, len(ranges), 3)],
                       fontsize=7)
    ax.set_yticks(range(0, len(bearings), 4))
    ax.set_yticklabels([f"{bearings[i]:.0f}" for i in range(0, len(bearings), 4)],
                       fontsize=7)
    ax.set_title(LABELS[method], fontsize=9, loc="left")
    return im


def figure(policy, h, tau, metric_key):
    metric = METRICS[metric_key]
    rows = load(policy, h, tau)
    if not rows:
        sys.exit(f"no rows for policy={policy} h={h} tau={tau} -- run grid.py first")

    vals = [float(r[metric["column"]]) for r in rows if r[metric["column"]] != ""]
    vmin, vmax = min(vals), max(vals)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), sharex=True, sharey=True)
    im = None
    for ax, m in zip(axes.ravel(), METHODS):
        im = panel(ax, rows, m, metric, vmin, vmax)
    for ax in axes[-1]:
        ax.set_xlabel("detection range (m)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("bearing (deg)", fontsize=8)

    n_caught = sum(int(r["captured"]) for r in rows)
    fig.suptitle(
        f"{metric['label']}\n{policy}, h = {h:g} s, "
        f"$\\tau$ = {tau:g} s, v_flee = 8 m/s (assumed)   "
        f"[x = captured, {n_caught} of {len(rows)}]",
        fontsize=9.5, x=0.02, ha="left")
    fig.colorbar(im, ax=axes, shrink=0.82, pad=0.02,
                 label=metric["label"])

    name = f"heat_{metric_key}_{policy}_h{h:g}_tau{tau:g}"
    os.makedirs(IMAGES, exist_ok=True)
    png = os.path.join(IMAGES, name + ".png")
    fig.savefig(png, dpi=170, bbox_inches="tight")
    fig.savefig(os.path.join(IMAGES, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return png


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="defer",
                    choices=("defer", "prepare", "forecast_forward"))
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--metric", default="margin", choices=sorted(METRICS))
    ap.add_argument("--all", action="store_true",
                    help="every policy x h x tau for the chosen metric")
    args = ap.parse_args(argv)

    if args.all:
        made = []
        for policy in ("defer", "prepare", "forecast_forward"):
            for h in (0.5, 1.0, 2.0):
                for tau in (0.1, 0.3, 0.5):
                    made.append(figure(policy, h, tau, args.metric))
        print(f"wrote {len(made)} figures to {IMAGES}")
    else:
        print("wrote", figure(args.policy, args.h, args.tau, args.metric))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.exit(main(sys.argv[1:]))
