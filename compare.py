#!/usr/bin/env python3
"""Compare methods, policies, and parameters against each other.

Four questions, four subcommands:

    python3 compare.py methods                 one policy, all four methods
    python3 compare.py policies                one method, all three policies
    python3 compare.py matrix                  every policy x method combination
    python3 compare.py sweep                   what h and tau do to each method

Each takes --h / --tau / --policy / --method to pin whatever it is not varying,
and --fig to also write a figure.  Tables need nothing but the standard
library; figures need matplotlib.

Reads Images/grid_results.csv and Images/grid_required_range.csv, so run
`python3 grid.py` first.

Every handover metric is computed over cases that actually committed.  Cases
caught before h expires never hand over, and counting their zeros would drag
the averages down.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(os.path.dirname(HERE), "Images")
CSV_PATH = os.path.join(IMAGES, "grid_results.csv")
REQ_PATH = os.path.join(IMAGES, "grid_required_range.csv")

METHODS = ("cold", "warm", "blend", "filtered")
POLICIES = ("prepare", "forecast_forward", "defer")
LABELS = {"cold": "cold", "warm": "warm",
          "blend": "blend 0.5 s", "filtered": "filtered 0.5 s"}
COLORS = {"cold": "#2E6FD9", "warm": "#C77A1E",
          "blend": "#1F8A70", "filtered": "#B5379A"}
#: Short forms, for grids where the full label would collide with its neighbour.
SHORT = {"cold": "cold", "warm": "warm", "blend": "blend", "filtered": "filtered"}

H_VALUES = (0.5, 1.0, 2.0)
TAU_VALUES = (0.1, 0.3, 0.5)

METRICS = {
    "margin":   ("worst margin (m)",      lambda s: s["worst_margin"],  "lower is worse"),
    "jolt":     ("mean cmd dv (m/s)",     lambda s: s["mean_dv"],       "lower is better"),
    "maxjolt":  ("max cmd dv (m/s)",      lambda s: s["max_dv"],        "lower is better"),
    "captures": ("captures after commit", lambda s: s["captures"],      "lower is better"),
    "falsecert":("false certificates",    lambda s: s["false_cert"],    "lower is better"),
    "reqrange": ("required range (m)",    lambda s: s["req_range"],     "lower is better"),
}


def load_rows():
    if not os.path.exists(CSV_PATH):
        sys.exit(f"no grid results at {CSV_PATH} -- run `python3 grid.py` first")
    rows = []
    with open(CSV_PATH) as fh:
        for r in csv.DictReader(fh):
            for k in ("h", "tau", "cmd_vel_jump", "cmd_accel_jump",
                      "detect_range", "bearing_deg", "turn_angle_at_commit"):
                r[k] = float(r[k])
            for k in ("captured", "committed", "captured_after_commit",
                      "false_certificate"):
                r[k] = int(r[k])
            m = r["min_margin_after_commit"]
            r["min_margin_after_commit"] = None if m == "" else float(m)
            rows.append(r)
    return rows


def load_req():
    out = {}
    if not os.path.exists(REQ_PATH):
        return out
    with open(REQ_PATH) as fh:
        for r in csv.DictReader(fh):
            v = r["required_range"]
            out[(r["policy"], float(r["h"]), float(r["tau"]), r["method"])] = (
                None if v == "" else float(v))
    return out


def stats(rows, req, policy, h, tau, method):
    cell = [r for r in rows if r["policy"] == policy and r["method"] == method
            and abs(r["h"] - h) < 1e-9 and abs(r["tau"] - tau) < 1e-9]
    sw = [r for r in cell if r["committed"]]
    margins = [r["min_margin_after_commit"] for r in sw
               if r["min_margin_after_commit"] is not None]
    dv = [r["cmd_vel_jump"] for r in sw]
    rr = req.get((policy, h, tau, method))
    return {
        "cases": len(cell),
        "committed": len(sw),
        "captures": sum(r["captured_after_commit"] for r in cell),
        "false_cert": sum(r["false_certificate"] for r in cell),
        "mean_dv": (sum(dv) / len(dv)) if dv else float("nan"),
        "max_dv": max(dv) if dv else float("nan"),
        "worst_margin": min(margins) if margins else float("nan"),
        "req_range": rr if rr is not None else float("nan"),
    }


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{nd}f}"


HEAD = (f"{'':<20}{'cases':>7}{'captures':>10}{'false cert':>12}"
        f"{'mean dv':>10}{'max dv':>9}{'worst margin':>14}{'req range':>11}")


def row_line(label, s):
    return (f"{label:<20}{s['committed']:>7}{s['captures']:>10}{s['false_cert']:>12}"
            f"{fmt(s['mean_dv'], 3):>10}{fmt(s['max_dv']):>9}"
            f"{fmt(s['worst_margin']):>14}{fmt(s['req_range'], 1):>11}")


def rule(n=93):
    print("-" * n)


def header(title, pinned):
    print()
    print(title)
    print("  " + "   ".join(f"{k} = {v}" for k, v in pinned.items()))
    rule()
    print(HEAD)
    rule()


# ---------------------------------------------------------------- subcommands

def cmd_methods(rows, req, args):
    header("METHODS COMPARED (policy and parameters held fixed)",
           {"policy": args.policy, "h": f"{args.h:g} s", "tau": f"{args.tau:g} s"})
    for m in METHODS:
        print(row_line(LABELS[m], stats(rows, req, args.policy, args.h, args.tau, m)))
    rule()
    print("Only the start-up method changes. Differences are the handover alone.")


def cmd_policies(rows, req, args):
    header("POLICIES COMPARED (method and parameters held fixed)",
           {"method": LABELS[args.method], "h": f"{args.h:g} s",
            "tau": f"{args.tau:g} s"})
    for p in POLICIES:
        print(row_line(p, stats(rows, req, p, args.h, args.tau, args.method)))
    rule()
    print("Only the wait policy changes -- what the drone does before the switch.")


def cmd_matrix(rows, req, args):
    label, pick, direction = METRICS[args.metric]
    print()
    print(f"EVERY POLICY x METHOD COMBINATION -- {label}  ({direction})")
    print(f"  h = {args.h:g} s   tau = {args.tau:g} s")
    rule(72)
    print(f"{'policy':<20}" + "".join(f"{SHORT[m]:>12}" for m in METHODS))
    rule(72)
    for p in POLICIES:
        cells = [pick(stats(rows, req, p, args.h, args.tau, m)) for m in METHODS]
        print(f"{p:<20}" + "".join(f"{fmt(c):>12}" for c in cells))
    rule(72)
    print("Rows change the wait policy, columns change the start-up method.")
    print("Read down a column for policy effect, across a row for method effect.")


def cmd_sweep(rows, req, args):
    label, pick, direction = METRICS[args.metric]
    for vary, values, other, other_val in (
            ("h", H_VALUES, "tau", args.tau),
            ("tau", TAU_VALUES, "h", args.h)):
        print()
        print(f"EFFECT OF {vary.upper()} -- {label}  ({direction})")
        print(f"  policy = {args.policy}   {other} = {other_val:g} s")
        rule(72)
        print(f"{'method':<20}" + "".join(f"{f'{vary}={v:g}':>12}" for v in values))
        rule(72)
        for m in METHODS:
            cells = []
            for v in values:
                h = v if vary == "h" else args.h
                tau = v if vary == "tau" else args.tau
                cells.append(pick(stats(rows, req, args.policy, h, tau, m)))
            print(f"{SHORT[m]:<20}" + "".join(f"{fmt(c):>12}" for c in cells))
        rule(72)


# ---------------------------------------------------------------- figures

def figure(rows, req, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 190, "savefig.bbox": "tight",
        "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
        "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 1.8,
    })
    label, pick, direction = METRICS[args.metric]
    os.makedirs(IMAGES, exist_ok=True)

    if args.cmd == "sweep":
        # One panel per policy, one line per method, x = the swept parameter.
        fig, axes = plt.subplots(2, len(POLICIES), figsize=(10.5, 6.0),
                                 sharex="row", sharey=True)
        for col, policy in enumerate(POLICIES):
            for row_i, (vary, values) in enumerate((("h", H_VALUES),
                                                    ("tau", TAU_VALUES))):
                ax = axes[row_i][col]
                for m in METHODS:
                    ys = []
                    for v in values:
                        h = v if vary == "h" else args.h
                        tau = v if vary == "tau" else args.tau
                        ys.append(pick(stats(rows, req, policy, h, tau, m)))
                    ax.plot(values, ys, color=COLORS[m], marker="o",
                            markersize=4, label=LABELS[m])
                ax.set_xticks(values)
                if row_i == 0:
                    ax.set_title(policy, fontsize=9.5, loc="left")
                ax.set_xlabel(f"{vary} (s)", fontsize=8)
        for r in range(2):
            axes[r][0].set_ylabel(label, fontsize=8)
        axes[0][0].legend(fontsize=7.5, loc="best")
        fig.suptitle(f"{label} against h (top) and tau (bottom)  --  {direction}",
                     fontsize=10, x=0.02, ha="left")
        name = f"cmp_sweep_{args.metric}"

    elif args.cmd == "matrix":
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        width = 0.2
        for i, m in enumerate(METHODS):
            xs = [j + (i - 1.5) * width for j in range(len(POLICIES))]
            ys = [pick(stats(rows, req, p, args.h, args.tau, m)) for p in POLICIES]
            ax.bar(xs, ys, width=width * 0.9, color=COLORS[m], label=LABELS[m])
        ax.set_xticks(range(len(POLICIES)))
        ax.set_xticklabels(POLICIES)
        ax.set_ylabel(label, fontsize=8)
        ax.legend(fontsize=7.5, ncol=4)
        ax.set_title(f"{label} by policy and method   "
                     f"(h = {args.h:g} s, tau = {args.tau:g} s)  --  {direction}",
                     fontsize=9.5, loc="left")
        name = f"cmp_matrix_{args.metric}_h{args.h:g}_tau{args.tau:g}"

    else:   # methods / policies -- a single grouped bar
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        if args.cmd == "methods":
            keys, ys = METHODS, [pick(stats(rows, req, args.policy, args.h,
                                            args.tau, m)) for m in METHODS]
            cols = [COLORS[m] for m in METHODS]
            names = [LABELS[m] for m in METHODS]
            title = f"{label} by method ({args.policy}, h = {args.h:g}, tau = {args.tau:g})"
            name = f"cmp_methods_{args.metric}_{args.policy}_h{args.h:g}_tau{args.tau:g}"
        else:
            keys, ys = POLICIES, [pick(stats(rows, req, p, args.h, args.tau,
                                             args.method)) for p in POLICIES]
            cols = ["#5A6472"] * len(POLICIES)
            names = list(POLICIES)
            title = (f"{label} by policy ({LABELS[args.method]}, "
                     f"h = {args.h:g}, tau = {args.tau:g})")
            name = f"cmp_policies_{args.metric}_{args.method}_h{args.h:g}_tau{args.tau:g}"
        ax.bar(range(len(keys)), ys, color=cols, width=0.6)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(title + f"  --  {direction}", fontsize=9.5, loc="left")

    png = os.path.join(IMAGES, name + ".png")
    fig.savefig(png)
    fig.savefig(os.path.join(IMAGES, name + ".pdf"))
    plt.close(fig)
    return png


def main(argv) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("methods", "policies", "matrix", "sweep"))
    ap.add_argument("--policy", default="defer", choices=POLICIES)
    ap.add_argument("--method", default="cold", choices=METHODS)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--metric", default="margin", choices=sorted(METRICS),
                    help="which number matrix and sweep report")
    ap.add_argument("--fig", action="store_true", help="also write a figure")
    args = ap.parse_args(argv)

    rows, req = load_rows(), load_req()
    {"methods": cmd_methods, "policies": cmd_policies,
     "matrix": cmd_matrix, "sweep": cmd_sweep}[args.cmd](rows, req, args)

    if args.fig:
        print("\nwrote", figure(rows, req, args))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.exit(main(sys.argv[1:]))
