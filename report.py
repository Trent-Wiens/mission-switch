#!/usr/bin/env python3
"""Validation against experiment 002, and safety-filter diagnostics.

    python3 report.py repro      experiment 002 comparison (step 1)
    python3 report.py filter     where the safety filter overrides the blend
    python3 report.py all

These are the two analyses `grid.py` does NOT do.

`repro` runs the LEGACY plant -- no speed loop, no speed cap -- because that is
what experiment 002 ran, and it is the only configuration in which its 11 m and
15/20/28 m figures mean anything. Everything else in the project uses the
realistic plant; this script is where the old physics lives.

`filter` sweeps detection range at one geometry to show where the safety filter
refuses the blend and where it lets it through. `grid.py` reports filtered as
just another method and `compare.py` aggregates over all geometries, so neither
answers "at what range does the filter stop intervening".

The start-up table that used to live here was superseded by `grid.py`, which
runs the same table over 432 geometries instead of 12 ranges head-on; the two
agreed to 0.01 m where they overlapped. The trace export was superseded by
`replay.py --csv`, which does it for any case rather than one hardcoded one.
"""

from __future__ import annotations

import csv
import math
import os
import sys

from pursuit_evader.params import Params
from pursuit_evader.presets import LEGACY, REALISTIC, V_FLEE
from pursuit_evader.scenarios import FROM_BEHIND, H_VALUES, HEAD_ON, TAU_VALUES
from pursuit_evader.sim import Scenario, run
from pursuit_evader.startup import blend, cold, filtered_blend, warm
from pursuit_evader.sweep import required_detection_range

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(os.path.dirname(HERE), "Images")



#: Numbers reported in the handoff, for comparison only.
REPORTED_002 = {"prepare": (11.0, 11.0, 11.0), "defer": (15.0, 20.0, 28.0)}


def _rule(width: int = 78) -> None:
    print("-" * width)


def _need(policy, h, bearing):
    return required_detection_range(
        bearing=bearing, policy=policy, h=h, startup=warm(), params=LEGACY
    )


def repro() -> None:
    """Step 1: required detection range, against what experiment 002 reported.

    Experiment 002's headline figures are the worst case over all bearings.
    The worst bearing is head-on, so that is what the first table reports.
    """
    print("\nEXPERIMENT 002 REPRODUCTION  (lag off, no speed cap, warm start)")
    print("Required detection range in metres, worst bearing (head-on).")
    _rule()
    print(f"{'policy':<18}" + "".join(f"h={h:<8g}" for h in H_VALUES) + "  reported 002")
    _rule()
    for policy in ("prepare", "forecast_forward", "defer", "stop_first"):
        cells = []
        for h in H_VALUES:
            r = _need(policy, h, HEAD_ON)
            cells.append("  n/a   " if r is None else f"{r:<8.1f}")
        reported = REPORTED_002.get(policy)
        tail = "  " + "/".join(f"{v:g}" for v in reported) if reported else ""
        print(f"{policy:<18}" + "".join(cells) + tail)
    _rule()
    print("prepare and defer reproduce the reported 11/11/11 and 15/20/28.")

    print("\nApproached from behind. Experiment 002 reported stop_first needing")
    print("1.9 to 4.3 times what prepare needs; this build gets less than that.")
    _rule()
    print(f"{'policy':<18}" + "".join(f"h={h:<8g}" for h in H_VALUES))
    _rule()
    for policy in ("prepare", "defer", "stop_first"):
        cells = []
        for h in H_VALUES:
            r = _need(policy, h, FROM_BEHIND)
            cells.append("  n/a   " if r is None else f"{r:<8.1f}")
        print(f"{policy:<18}" + "".join(cells))
    ratios = [
        _need("stop_first", h, FROM_BEHIND) / _need("prepare", h, FROM_BEHIND)
        for h in H_VALUES
    ]
    print(f"{'stop_first/prepare':<18}" + "".join(f"{r:<8.2f}" for r in ratios))
    _rule()
    print("Note: from behind, fleeing straight ahead is what surveying already")
    print("does, so defer and prepare coincide there. The gap between them only")
    print("opens at bearings where escaping means turning.")


def filter_engagement(h: float = 1.0, tau: float = 0.3) -> None:
    """Where the safety filter overrides the blend, and where it lets it run.

    This is the mechanism behind the stretch goal: if the filtered blend keeps
    the jolt down at ranges where a cold start is refused, that is the result.
    """
    p = REALISTIC.with_(tau=tau)
    print(f"\nSAFETY FILTER ENGAGEMENT  (tau = {tau} s, h = {h} s, head-on)")
    _rule(72)
    print(f"{'range (m)':>10}{'cold dv':>10}{'blend dv':>10}{'filtered dv':>13}"
          f"{'filtered?':>11}{'escapes':>9}")
    _rule(72)
    for rng in (14.0, 16.0, 18.0, 20.0, 24.0, 30.0, 40.0, 55.0):
        common = dict(detect_range=rng, bearing=HEAD_ON, h=h,
                      policy="defer", params=p, v_flee=V_FLEE)
        c = run(Scenario(startup=cold(), **common))
        b = run(Scenario(startup=blend(0.5), **common))
        f = run(Scenario(startup=filtered_blend(0.5), **common))
        overridden = "yes" if abs(f.cmd_vel_jump - b.cmd_vel_jump) > 1e-6 else "no"
        print(f"{rng:>10.0f}{c.cmd_vel_jump:>10.2f}{b.cmd_vel_jump:>10.2f}"
              f"{f.cmd_vel_jump:>13.2f}{overridden:>11}{str(not f.captured):>9}")
    _rule(72)
    print("dv = jump in commanded velocity at commit (m/s); lower is gentler.")
    print("Where 'filtered?' is yes, the filter refused the blend and took the")
    print("full escape command instead, so it reduces to a cold start.")


def main(argv: list[str]) -> int:
    what = argv[0] if argv else "all"
    if what not in ("repro", "filter", "all"):
        print(__doc__)
        return 2
    if what in ("repro", "all"):
        repro()
    if what in ("filter", "all"):
        filter_engagement()
    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.exit(main(sys.argv[1:]))
