"""Physical parameters of the pursuit/evasion toy.

Defaults are the numbers given in the project 2 handoff (Notes/).  Anything not
fixed by the handoff is marked OPEN below and should be settled before the
numbers in section 4 of the handoff are treated as results.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class Params:
    # --- evader (our drone) -------------------------------------------------
    surveil_radius: float = 30.0      # m, radius of the surveillance circle
    surveil_speed: float = 3.0        # m/s, speed while surveying
    geofence_radius: float = 80.0     # m, boundary that must never be crossed
    a_max: float = 4.0                # m/s^2, acceleration cap

    # OPEN: the handoff never states an evader speed cap.  `None` reproduces
    # experiment 002, where the evader accelerates away without limit and the
    # certificate's braking term can be driven to zero.  The audit lists "no
    # speed cap" as a gap to close, so this is the knob that closes it.
    v_max: Optional[float] = None     # m/s, or None for uncapped

    # --- pursuer ------------------------------------------------------------
    v_pursuer: float = 5.0            # m/s, pursuer speed (constant, pure pursuit)
    capture_radius: float = 3.0       # m

    # --- speed loop (step 3 of the handoff) ---------------------------------
    lag: bool = True                  # the switch that turns the lag off
    tau: float = 0.3                  # s, first-order speed-loop time constant

    # --- integration --------------------------------------------------------
    dt: float = 0.005                 # s, fixed step

    # --- evaluation ---------------------------------------------------------
    # One window for every policy, measured from detection.  The audit's second
    # caveat was that policies had been scored over different windows.
    t_eval: float = 20.0              # s after detection

    def with_(self, **kwargs) -> "Params":
        """Return a copy with the given fields replaced."""
        return replace(self, **kwargs)

    def __post_init__(self) -> None:
        if self.tau <= 0.0:
            raise ValueError("tau must be positive; use lag=False to disable lag")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.lag and self.dt > self.tau / 4.0:
            raise ValueError(
                f"dt={self.dt} is too coarse for tau={self.tau}; "
                f"use dt <= {self.tau / 4.0}"
            )


DEFAULT = Params()
