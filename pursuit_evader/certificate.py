"""The safety certificate and the estimator that feeds it.

The rule from the handoff:

    margin = d - rho - r_capture - max(0, v_p_max - v_r)^2 / (2 a_max)   >= 0

`d` is the *estimated* distance to the pursuer, `rho` pays for the estimate
being slightly old, and `v_r` is the evader's speed directly away from the
pursuer.  The hidden assumption is that the evader gets full acceleration the
instant it asks.  `extra` is the slot where the start-up cost derived in step 8
of the handoff gets subtracted; it is 0.0 until that term exists.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

from .params import Params
from .vec import Vec2


@dataclass(frozen=True)
class Estimate:
    """What the controller believes about the pursuer, and how old it is."""

    pos: Vec2
    vel: Vec2
    age: float          # true age of this estimate, seconds


def rho_allowance(assumed_age: float, evader_speed: float, params: Params) -> float:
    """Distance the gap could have closed since the estimate was taken.

    Worst case the pursuer flew straight in while the evader flew straight out
    of its own retreat, so the gap shrinks at up to (v_p + v_e).  If
    `assumed_age` understates the true age, this is too small and the
    certificate can be satisfied while the drone is not actually safe -- the
    false-certificate mechanism experiment 002 found.
    """
    return (params.v_pursuer + evader_speed) * assumed_age


def margin(
    d: float,
    v_r: float,
    rho: float,
    params: Params,
    extra: float = 0.0,
) -> float:
    """Safety margin in metres. Non-negative means the certificate holds."""
    closing = max(0.0, params.v_pursuer - v_r)
    braking = closing * closing / (2.0 * params.a_max)
    return d - rho - params.capture_radius - braking - extra


def margin_from_state(
    evader_pos: Vec2,
    evader_vel: Vec2,
    pursuer_pos: Vec2,
    params: Params,
    assumed_age: float = 0.0,
    extra: float = 0.0,
) -> float:
    """Convenience wrapper: build the certificate inputs from raw states."""
    rel = evader_pos - pursuer_pos
    d = rel.norm
    v_r = evader_vel.dot(rel.unit())
    rho = rho_allowance(assumed_age, evader_vel.norm, params)
    return margin(d, v_r, rho, params, extra)


class DelayedEstimator:
    """Hands the controller a pursuer state from `true_age` seconds ago.

    `assumed_age` is what the certificate budgets for via rho.  Setting
    `true_age > assumed_age` reproduces the experiment 002 finding that an
    estimate one second staler than assumed produces false certificates and
    captures.
    """

    def __init__(
        self,
        true_age: float = 0.0,
        assumed_age: Optional[float] = None,
        dt: float = 0.005,
    ) -> None:
        if true_age < 0.0:
            raise ValueError("true_age must be non-negative")
        self.true_age = true_age
        self.assumed_age = true_age if assumed_age is None else assumed_age
        self.dt = dt
        self._buffer: Deque[Tuple[Vec2, Vec2]] = deque()
        self._depth = max(1, int(round(true_age / dt)) + 1)

    def update(self, pursuer_pos: Vec2, pursuer_vel: Vec2) -> None:
        self._buffer.append((pursuer_pos, pursuer_vel))
        while len(self._buffer) > self._depth:
            self._buffer.popleft()

    def estimate(self) -> Estimate:
        if not self._buffer:
            raise RuntimeError("estimator has no measurements yet; call update() first")
        pos, vel = self._buffer[0]
        age = (len(self._buffer) - 1) * self.dt
        return Estimate(pos=pos, vel=vel, age=age)
