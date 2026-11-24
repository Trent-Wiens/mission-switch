#!/usr/bin/env python3
"""Static figures for the write-up.

    python3 plot.py            all figures -> ../Images/
    python3 plot.py switch     only the milestone figure

Needs matplotlib. The simulator itself stays standard-library only; plotting is
separate tooling on purpose, so `run_tests.py` never depends on it.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pursuit_evader.params import Params
from pursuit_evader.presets import V_FLEE
from pursuit_evader.scenarios import HEAD_ON
from pursuit_evader.sim import Scenario, run
from pursuit_evader.startup import blend, cold, filtered_blend, warm

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(os.path.dirname(HERE), "Images")

#: Same validated categorical palette as the interactive viewer, so a figure in
#: the paper and a screenshot of the viewer agree on which colour means what.
COLORS = {
    "cold": "#2E6FD9",
    "warm": "#C77A1E",
    "blend": "#1F8A70",
    "filtered": "#B5379A",
}
CRITICAL = "#D0342C"

METHODS = [
    ("cold", "Cold start", cold),
    ("warm", "Warm start", warm),
    ("blend", "Blend 0.5 s", lambda: blend(0.5)),
    ("filtered", "Filtered 0.5 s", lambda: filtered_blend(0.5)),
]

H = 1.0
TAU = 0.3
DETECT = 25.0


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
    })


def runs(detect_range=DETECT, tau=TAU, window=8.0):
    p = Params(lag=True, tau=tau, dt=0.005, v_max=V_FLEE, t_eval=window)
    out = {}
    for key, label, make in METHODS:
        out[key] = (label, run(Scenario(
            detect_range=detect_range, bearing=HEAD_ON, h=H, policy="defer",
            startup=make(), params=p, v_flee=V_FLEE,
        )))
    return out


def mark_commit(ax) -> None:
    ax.axvline(H, color="0.45", linestyle="--", linewidth=0.9, zorder=1)
    ax.annotate("commit", xy=(H, ax.get_ylim()[1]), xytext=(3, -3),
                textcoords="offset points", va="top", ha="left",
                fontsize=7.5, color="0.4")


def fig_switch(data) -> str:
    """The two-week milestone: commanded and actual velocity across the switch,
    head-on at h = 1 s, all four methods on one plot."""
    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    for key, _, _ in METHODS:
        label, r = data[key]
        t = [s.t for s in r.trace]
        ax.plot(t, [s.cmd.norm for s in r.trace], color=COLORS[key],
                linestyle=":", linewidth=1.3, zorder=2)
        ax.plot(t, [s.evader_vel.norm for s in r.trace], color=COLORS[key],
                label=label, zorder=3)

    ax.set_xlabel("time since detection (s)")
    ax.set_ylabel("speed (m/s)")
    # Everything of interest happens in the first few seconds; the rest is the
    # drone sitting at its flee speed.
    ax.set_xlim(0, 5.0)
    ax.set_ylim(bottom=0)
    mark_commit(ax)

    handles, labels = ax.get_legend_handles_labels()
    style_note = plt.Line2D([], [], color="0.3", linestyle=":", linewidth=1.3,
                            label="commanded (dotted)")
    actual_note = plt.Line2D([], [], color="0.3", linewidth=1.8,
                             label="actual (solid)")
    ax.legend(handles + [actual_note, style_note],
              labels + ["actual (solid)", "commanded (dotted)"],
              loc="lower right", ncol=2, fontsize=8)

    ax.set_title(
        "Commanded vs actual speed across the handover\n"
        f"head-on, detection at {DETECT:g} m, h = {H:g} s, "
        f"$\\tau$ = {TAU:g} s, flee speed {V_FLEE:g} m/s",
        fontsize=9.5, loc="left")

    return save(fig, "fig_switch_speed")


def fig_switch_panels(data) -> str:
    """Same data as small multiples, which reads better when the traces cross."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True, sharey=True)

    for ax, (key, _, _) in zip(axes.ravel(), METHODS):
        label, r = data[key]
        t = [s.t for s in r.trace]
        ax.plot(t, [s.cmd.norm for s in r.trace], color=COLORS[key],
                linestyle=":", linewidth=1.3, label="commanded")
        ax.plot(t, [s.evader_vel.norm for s in r.trace], color=COLORS[key],
                label="actual")
        ax.axvline(H, color="0.45", linestyle="--", linewidth=0.9)
        ax.set_title(label + f"   $\\Delta v$ = {r.cmd_vel_jump:.2f} m/s",
                     fontsize=8.5, loc="left")
        ax.set_ylim(bottom=0)

    for ax in axes[-1]:
        ax.set_xlabel("time since detection (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("speed (m/s)")
    axes[0, 0].legend(loc="lower right", fontsize=7.5)

    fig.suptitle("Handover, method by method", fontsize=9.5, x=0.09, ha="left")
    return save(fig, "fig_switch_panels")


def fig_margin(data) -> str:
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for key, _, _ in METHODS:
        label, r = data[key]
        ax.plot([s.t for s in r.trace], [s.margin for s in r.trace],
                color=COLORS[key], label=label)
    ax.axhline(0, color=CRITICAL, linestyle=":", linewidth=1.1)
    ax.annotate("certificate fails below 0", xy=(0.02, 0), xytext=(2, 4),
                textcoords="offset points", fontsize=7.5, color=CRITICAL)
    ax.set_xlabel("time since detection (s)")
    ax.set_ylabel("margin (m)")
    ax.set_xlim(0, max(s.t for _, r in data.values() for s in r.trace))
    mark_commit(ax)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.set_title("Safety margin across the handover", fontsize=9.5, loc="left")
    return save(fig, "fig_margin")


def fig_tracks(data) -> str:
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    cfg = Params()

    ax.add_patch(plt.Circle((0, 0), cfg.surveil_radius, fill=False,
                            edgecolor="0.6", linestyle=":", linewidth=1.0))
    ax.add_patch(plt.Circle((0, 0), cfg.geofence_radius, fill=False,
                            edgecolor="0.6", linestyle="--", linewidth=1.0))

    for key, _, _ in METHODS:
        label, r = data[key]
        ax.plot([s.evader_pos.x for s in r.trace],
                [s.evader_pos.y for s in r.trace],
                color=COLORS[key], label=label)

    label, r0 = data["cold"]
    ax.plot([s.pursuer_pos.x for s in r0.trace],
            [s.pursuer_pos.y for s in r0.trace],
            color="0.35", linestyle="--", linewidth=1.3, label="pursuer")
    ax.plot(r0.trace[0].pursuer_pos.x, r0.trace[0].pursuer_pos.y,
            marker="v", color="0.2", markersize=6)

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="lower left", fontsize=7.5)
    ax.set_title("Tracks over the evaluation window", fontsize=9.5, loc="left")
    return save(fig, "fig_tracks")


def fig_tau_sweep() -> str:
    """The demanded-acceleration jump scales as 1/tau."""
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    taus = [0.1, 0.2, 0.3, 0.4, 0.5]

    for key, label, make in METHODS:
        jumps = []
        for tau in taus:
            p = Params(lag=True, tau=tau, dt=min(0.005, tau / 40.0),
                       v_max=V_FLEE, t_eval=8.0)
            r = run(Scenario(detect_range=DETECT, bearing=HEAD_ON, h=H,
                             policy="defer", startup=make(), params=p,
                             v_flee=V_FLEE))
            jumps.append(r.cmd_accel_jump)
        ax.plot(taus, jumps, color=COLORS[key], marker="o", markersize=4,
                label=label)

    ax.set_xlabel(r"speed-loop time constant $\tau$ (s)")
    ax.set_ylabel(r"demanded accel jump (m/s$^2$)")
    ax.legend(fontsize=8)
    ax.set_title(r"Jolt scales as $1/\tau$, as $(v_{cmd}-v)/\tau$ predicts",
                 fontsize=9.5, loc="left")
    return save(fig, "fig_tau_sweep")


def save(fig, name: str) -> str:
    os.makedirs(IMAGES, exist_ok=True)
    png = os.path.join(IMAGES, name + ".png")
    fig.savefig(png)
    fig.savefig(os.path.join(IMAGES, name + ".pdf"))
    plt.close(fig)
    return png


def main(argv) -> int:
    style()
    what = argv[0] if argv else "all"
    data = runs()
    made = []
    if what in ("switch", "all"):
        made.append(fig_switch(data))
        made.append(fig_switch_panels(data))
    if what in ("margin", "all"):
        made.append(fig_margin(data))
    if what in ("tracks", "all"):
        made.append(fig_tracks(data))
    if what in ("tau", "all"):
        made.append(fig_tau_sweep())
    for m in made:
        print("wrote", m, "(+ .pdf)")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.exit(main(sys.argv[1:]))
