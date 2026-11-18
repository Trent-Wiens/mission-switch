"""Pursuit/evader toy: starting an escape controller without a jolt.

Standard library only, by design (see Notes/trent_project_2_handoff.md).
"""

from .certificate import DelayedEstimator, Estimate, margin, margin_from_state
from .controllers import (
    EscapeController,
    HoverController,
    PrepareController,
    SurveillanceController,
)
from .dynamics import Evader, Pursuer
from .params import DEFAULT, Params
from .sim import Result, Sample, Scenario, run
from .startup import StartupMethod, blend, cold, filtered_blend, warm
from .vec import Vec2, from_polar

__all__ = [
    "DEFAULT",
    "DelayedEstimator",
    "Estimate",
    "Evader",
    "EscapeController",
    "HoverController",
    "Params",
    "PrepareController",
    "Pursuer",
    "Result",
    "Sample",
    "Scenario",
    "StartupMethod",
    "SurveillanceController",
    "Vec2",
    "blend",
    "cold",
    "filtered_blend",
    "from_polar",
    "margin",
    "margin_from_state",
    "run",
    "warm",
]
