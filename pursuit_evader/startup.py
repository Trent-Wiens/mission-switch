"""The four start-up methods from step 6 of the handoff.

    cold      escape controller's memory starts at zero
    warm      memory seeded from the measured velocity
    blend     cold memory, output ramped from the outgoing command over T_b
    filtered  blend, then passed through a backup-check safety filter

`blend` and `filtered` keep the *cold* memory on purpose, so the blend's effect
is isolated from the warm-start effect.  `warm_memory=True` combines them; the
handoff treats them as four separate methods, so the presets below do too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .certificate import Estimate, margin_from_state
from .controllers import EscapeController
from .dynamics import Evader, Pursuer
from .params import Params
from .vec import ZERO, Vec2


def default_horizon(params: Params) -> float:
    """How far ahead the backup check has to look.

    The certificate claims safety *forever*, but a finite prediction only
    checks a window, so the window has to be long enough to reach the point
    where the margin stops falling.  Fleeing at full acceleration, that is when
    the retreat speed reaches the pursuer's speed: starting from a worst-case
    `-surveil_speed` of retreat speed, that takes
    (v_pursuer + surveil_speed) / a_max seconds.  Beyond it the pursuer never
    closes again, so nothing worse can appear later.

    The speed loop delays reaching that point, hence the 4*tau floor.
    """
    settle = (params.v_pursuer + params.surveil_speed) / params.a_max
    return max(4.0 * params.tau, settle)


def predict_margin(
    evader: Evader,
    est: Estimate,
    v_cmd: Vec2,
    params: Params,
    horizon: float,
    substeps: int = 20,
) -> float:
    """Smallest margin seen over `horizon` if `v_cmd` is held and the pursuer
    flies pure pursuit.  This is the "predict a fallback manoeuvre and check it"
    structure of the toy's s2_filter, extended to account for the speed loop.
    """
    h = horizon / substeps
    pos, vel = evader.pos, evader.vel
    p_pos = est.pos
    worst = float("inf")

    for _ in range(substeps):
        if params.lag:
            a = (v_cmd - vel) / params.tau
        else:
            a = (v_cmd - vel) / h
        a = a.clamp_norm(params.a_max)
        vel = vel + a * h
        if params.v_max is not None:
            vel = vel.clamp_norm(params.v_max)
        pos = pos + vel * h
        p_pos = p_pos + (pos - p_pos).unit() * params.v_pursuer * h

        m = margin_from_state(pos, vel, p_pos, params, assumed_age=0.0)
        worst = min(worst, m)

    return worst


def safety_filter(
    candidate: Vec2,
    fallback: Vec2,
    evader: Evader,
    est: Estimate,
    params: Params,
    horizon: Optional[float] = None,
    tolerance: float = 1e-3,
    max_iter: int = 24,
) -> Vec2:
    """Move `candidate` toward `fallback` as little as possible, so that the
    predicted margin stays non-negative.

    Minimal deviation is searched along the segment candidate->fallback, which
    is a one-dimensional stand-in for the QP a proper CBF filter would solve.
    """
    if horizon is None:
        horizon = default_horizon(params)

    if predict_margin(evader, est, candidate, params, horizon) >= 0.0:
        return candidate
    if predict_margin(evader, est, fallback, params, horizon) < 0.0:
        # Even the fallback cannot certify. Take it anyway; it is the best move.
        return fallback

    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        if hi - lo < tolerance:
            break
        mid = 0.5 * (lo + hi)
        trial = candidate + (fallback - candidate) * mid
        if predict_margin(evader, est, trial, params, horizon) >= 0.0:
            hi = mid
        else:
            lo = mid
    return candidate + (fallback - candidate) * hi


@dataclass
class StartupMethod:
    """How the escape controller is brought online at the commit instant."""

    name: str
    warm_memory: bool = False
    blend_time: float = 0.0         # T_b, seconds; 0 disables the blend
    filtered: bool = False

    # Set at commit time.
    _commit_cmd: Vec2 = field(default=ZERO, repr=False)
    _committed: bool = field(default=False, repr=False)

    def on_commit(self, evader: Evader, escape: EscapeController, last_cmd: Vec2) -> None:
        """Hand over. `last_cmd` is what the outgoing controller just sent."""
        escape.reset(evader, evader.vel if self.warm_memory else None)
        self._commit_cmd = last_cmd
        self._committed = True

    def command(
        self,
        t_since_commit: float,
        escape_cmd: Vec2,
        outgoing_cmd: Vec2,
        evader: Evader,
        est: Estimate,
        params: Params,
    ) -> Vec2:
        """Mix the escape command with the outgoing one and optionally filter."""
        if self.blend_time > 0.0 and t_since_commit < self.blend_time:
            s = t_since_commit / self.blend_time
            cmd = outgoing_cmd * (1.0 - s) + escape_cmd * s
        else:
            cmd = escape_cmd

        if self.filtered:
            cmd = safety_filter(cmd, escape_cmd, evader, est, params)
        return cmd


def cold() -> StartupMethod:
    return StartupMethod(name="cold")


def warm() -> StartupMethod:
    return StartupMethod(name="warm", warm_memory=True)


def blend(blend_time: float = 0.5) -> StartupMethod:
    return StartupMethod(name=f"blend_{blend_time:g}s", blend_time=blend_time)


def filtered_blend(blend_time: float = 0.5) -> StartupMethod:
    return StartupMethod(
        name=f"filtered_{blend_time:g}s", blend_time=blend_time, filtered=True
    )


#: The four methods the results table in section 4 of the handoff needs.
METHODS = {
    "cold": cold,
    "warm": warm,
    "blend": blend,
    "filtered": filtered_blend,
}

#: Blend lengths the handoff asks to sweep.
BLEND_TIMES = (0.25, 0.5, 1.0)
