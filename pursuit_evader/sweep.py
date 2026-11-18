"""Grid runs and the headline number: required detection range.

`required_detection_range` is the smallest warning distance at which the drone
still escapes.  Experiment 002 reported 11 m for `prepare` and 15/20/28 m for
`defer`; reproducing that ordering is the step 1 acceptance check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .params import DEFAULT, Params
from .scenarios import bearings, ranges
from .sim import Result, Scenario, run
from .startup import StartupMethod, cold


def escapes(scenario: Scenario) -> bool:
    return not run(scenario).captured


def required_detection_range(
    bearing: float,
    lo: float = 1.0,
    hi: float = 60.0,
    tolerance: float = 0.25,
    **scenario_kwargs,
) -> Optional[float]:
    """Smallest detection range at which the drone escapes, by bisection.

    Returns None if even `hi` is not enough.  Assumes escape is monotone in
    range, which holds for these dynamics but is worth re-checking whenever the
    controllers change -- `verify_monotone` below does that.
    """
    if not escapes(Scenario(detect_range=hi, bearing=bearing, **scenario_kwargs)):
        return None
    if escapes(Scenario(detect_range=lo, bearing=bearing, **scenario_kwargs)):
        return lo

    while hi - lo > tolerance:
        mid = 0.5 * (lo + hi)
        if escapes(Scenario(detect_range=mid, bearing=bearing, **scenario_kwargs)):
            hi = mid
        else:
            lo = mid
    return hi


def verify_monotone(bearing: float, n: int = 20, **scenario_kwargs) -> bool:
    """Check that escaping is monotone in detection range at this bearing."""
    outcomes = [
        escapes(Scenario(detect_range=r, bearing=bearing, **scenario_kwargs))
        for r in ranges(n, 2.0, 50.0)
    ]
    seen_escape = False
    for ok in outcomes:
        if ok:
            seen_escape = True
        elif seen_escape:
            return False
    return True


def worst_required_range(**scenario_kwargs) -> Tuple[float, Optional[float]]:
    """Over all bearings, the one that needs the most warning."""
    worst_bearing, worst = 0.0, -1.0
    for b in bearings():
        r = required_detection_range(bearing=b, **scenario_kwargs)
        if r is None:
            return b, None
        if r > worst:
            worst_bearing, worst = b, r
    return worst_bearing, worst


@dataclass
class GridSummary:
    n_cases: int = 0
    captures: int = 0
    captures_after_commit: int = 0
    false_certificates: int = 0
    geofence_breaches: int = 0
    peak_cmd_accel: float = 0.0
    peak_cmd_vel_jump: float = 0.0
    min_margin_after_commit: float = float("inf")

    def add(self, r: Result) -> None:
        self.n_cases += 1
        self.captures += int(r.captured)
        self.captures_after_commit += int(r.captured_after_commit)
        self.false_certificates += int(r.false_certificate)
        self.geofence_breaches += int(r.geofence_breach)
        self.peak_cmd_accel = max(self.peak_cmd_accel, r.peak_cmd_accel)
        self.peak_cmd_vel_jump = max(self.peak_cmd_vel_jump, r.cmd_vel_jump)
        if not math.isnan(r.min_margin_after_commit):
            self.min_margin_after_commit = min(
                self.min_margin_after_commit, r.min_margin_after_commit
            )


def summarize(scenarios: Iterable[Scenario]) -> GridSummary:
    summary = GridSummary()
    for s in scenarios:
        summary.add(run(s))
    return summary
