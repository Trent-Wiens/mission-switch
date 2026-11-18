#!/usr/bin/env python3
"""Regenerate the numbers. One command per section of the handoff.

    python3 report.py repro      experiment 002 comparison (step 1)
    python3 report.py startup    the four start-up methods (step 6)
    python3 report.py filter     where the safety filter overrides the blend
    python3 report.py trace      commanded/actual velocity across the switch
    python3 report.py all

`trace` writes CSV to Images/ so the milestone figure can be plotted from it
without adding a plotting dependency to the simulator itself.
"""

from __future__ import annotations

import csv
import math
import os
import sys

from pursuit_evader.params import Params
from pursuit_evader.scenarios import FROM_BEHIND, H_VALUES, HEAD_ON, TAU_VALUES
from pursuit_evader.sim import Scenario, run
from pursuit_evader.startup import blend, cold, filtered_blend, warm
from pursuit_evader.sweep import required_detection_range

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(os.path.dirname(HERE), "Images")

#: Experiment 002's plant: full immediate authority, no speed cap.
LEGACY = Params(lag=False, v_max=None, dt=0.005, t_eval=25.0)

#: OPEN QUESTION -- the evader's top speed.
#:
#: The handoff never states one.  It matters more than it looks:
#:
#:   v_max = None  the drone outruns the pursuer and escapes for good.  This is
#:                 what experiment 002 did.  But the escape command then has to
#:                 be a huge saturating velocity, which pins the acceleration at
#:                 a_max regardless of tau -- so the speed loop does nothing and
#:                 the whole lag study has no effect to measure.
#:
#:   v_max < 5     matches the handoff's prose ("the pursuer can outrun the drone
#:                 on the straight"), but then the retreat speed never reaches the
#:                 pursuer's, the certificate's braking term never vanishes, and
#:                 "safe forever" is false by construction -- the drone only ever
#:                 buys time.
#:
#: V_FLEE below splits the difference so the lag has something to act on.  It is
#: an assumption, not a result, and it needs to be settled before the section 4
#: table means anything.
V_FLEE = 8.0
REALISTIC = Params(lag=True, tau=0.3, v_max=V_FLEE, dt=0.005, t_eval=25.0)

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


def _methods(T_b: float = 0.5):
    return [cold(), warm(), blend(T_b), filtered_blend(T_b)]


def startup(tau: float = 0.3) -> None:
    """Step 6/7: the results table from section 4, at one tau."""
    p = REALISTIC.with_(tau=tau)
    print(f"\nSTART-UP METHODS  (tau = {tau} s, v_flee = {V_FLEE} m/s, head-on,")
    print("defer during the wait).  Worst case over detection ranges 12-34 m.")
    _rule(96)
    print(
        f"{'method':<16}{'h':>5}{'cmd dv':>9}{'cmd da':>9}{'peak cmd a':>12}"
        f"{'min margin':>12}{'captures':>10}{'false cert':>12}{'req range':>11}"
    )
    print(f"{'':<16}{'(s)':>5}{'(m/s)':>9}{'(m/s^2)':>9}{'(m/s^2)':>12}"
          f"{'(m)':>12}{'':>10}{'':>12}{'(m)':>11}")
    _rule(96)

    for method_factory in (cold, warm, lambda: blend(0.5), lambda: filtered_blend(0.5)):
        for h in H_VALUES:
            common = dict(bearing=HEAD_ON, h=h, policy="defer",
                          params=p, v_flee=V_FLEE)
            ranges = [12.0 + 2.0 * k for k in range(12)]
            results = [
                run(Scenario(detect_range=r, startup=method_factory(), **common))
                for r in ranges
            ]
            name = method_factory().name
            captures = sum(r.captured_after_commit for r in results)
            false_certs = sum(r.false_certificate for r in results)
            dv = max(r.cmd_vel_jump for r in results)
            da = max(r.cmd_accel_jump for r in results)
            acc = max(r.peak_cmd_accel for r in results)
            margins = [r.min_margin_after_commit for r in results
                       if not math.isnan(r.min_margin_after_commit)]
            worst_margin = min(margins) if margins else float("nan")
            req = required_detection_range(
                bearing=HEAD_ON, policy="defer", h=h,
                startup=method_factory(), params=p, v_flee=V_FLEE,
            )
            req_s = "n/a" if req is None else f"{req:.1f}"
            print(
                f"{name:<16}{h:>5g}{dv:>9.2f}{da:>9.1f}{acc:>12.1f}"
                f"{worst_margin:>12.2f}{captures:>10}{false_certs:>12}{req_s:>11}"
            )
        _rule(96)

    print("cmd dv     = |v_cmd just after commit - v_cmd just before|")
    print("cmd da     = change in demanded acceleration (v_cmd - v)/tau at commit")
    print("peak cmd a = largest demanded acceleration anywhere after commit")
    print("min margin = smallest certificate margin after commit, over the sweep")
    print("req range  = smallest detection range that still escapes (bisection)")


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


def trace(detect_range: float = 25.0, h: float = 1.0, tau: float = 0.3) -> None:
    """The 2-week milestone: commanded and actual velocity across the switch."""
    p = REALISTIC.with_(tau=tau, t_eval=8.0)
    os.makedirs(IMAGES, exist_ok=True)
    path = os.path.join(IMAGES, f"switch_trace_h{h:g}_tau{tau:g}.csv")

    rows = []
    for method in _methods():
        r = run(Scenario(detect_range=detect_range, bearing=HEAD_ON, h=h,
                         policy="defer", startup=method, params=p, v_flee=V_FLEE))
        for s in r.trace:
            rows.append({
                "method": method.name,
                "t": f"{s.t:.4f}",
                "t_since_commit": f"{s.t - h:.4f}",
                "cmd_speed": f"{s.cmd.norm:.5f}",
                "actual_speed": f"{s.evader_vel.norm:.5f}",
                "cmd_x": f"{s.cmd.x:.5f}", "cmd_y": f"{s.cmd.y:.5f}",
                "vel_x": f"{s.evader_vel.x:.5f}", "vel_y": f"{s.evader_vel.y:.5f}",
                "accel": f"{s.evader_accel.norm:.5f}",
                "distance": f"{s.distance:.4f}",
                "margin": f"{s.margin:.4f}",
                "committed": int(s.committed),
            })

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSWITCH TRACE  (head-on, detect at {detect_range} m, h = {h} s, tau = {tau} s)")
    print(f"wrote {len(rows)} rows to {path}")
    _rule()
    print(f"{'method':<16}{'cmd dv (m/s)':>14}{'peak cmd a':>13}{'captured':>10}")
    _rule()
    for method in _methods():
        r = run(Scenario(detect_range=detect_range, bearing=HEAD_ON, h=h,
                         policy="defer", startup=method, params=p, v_flee=V_FLEE))
        print(f"{method.name:<16}{r.cmd_vel_jump:>14.2f}"
              f"{r.peak_cmd_accel:>13.1f}{str(r.captured):>10}")
    _rule()


def main(argv: list[str]) -> int:
    what = argv[0] if argv else "all"
    if what in ("repro", "all"):
        repro()
    if what in ("startup", "all"):
        for tau in TAU_VALUES if what == "all" else (0.3,):
            startup(tau)
    if what in ("filter", "all"):
        filter_engagement()
    if what in ("trace", "all"):
        trace()
    if what not in ("repro", "startup", "filter", "trace", "all"):
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.exit(main(sys.argv[1:]))
