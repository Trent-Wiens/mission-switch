"""Controllers. Every controller emits a *velocity* command.

`EscapeController` carries the memory the whole project is about: a first-order
smoother on its own output, standing in for the filters/integrators/last-command
state a real controller holds.  Its initial value is what "cold start" versus
"warm start" actually means:

    cold start   smoother begins at zero        -> first command is a brake
    warm start   smoother begins at v_measured  -> first command matches motion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .certificate import Estimate
from .dynamics import Evader
from .params import Params
from .vec import ZERO, Vec2


class Controller:
    """Interface: return a velocity command for this step."""

    def command(self, t: float, evader: Evader, est: Estimate, dt: float) -> Vec2:
        raise NotImplementedError

    def reset(self, evader: Evader) -> None:
        pass


@dataclass
class SurveillanceController(Controller):
    """Hold a circle of radius `surveil_radius` at `surveil_speed`."""

    params: Params
    center: Vec2 = ZERO
    k_radial: float = 1.0           # 1/s, pull back onto the circle
    counter_clockwise: bool = True

    def command(self, t: float, evader: Evader, est: Estimate, dt: float) -> Vec2:
        p = self.params
        radial = evader.pos - self.center
        r = radial.norm
        if r == 0.0:
            # Degenerate: any direction will do, pick +x.
            return Vec2(p.surveil_speed, 0.0)

        radial_hat = radial.unit()
        tangent = radial_hat.perp()
        if not self.counter_clockwise:
            tangent = -tangent

        # Tangential motion plus a correction that drives r -> surveil_radius.
        desired = tangent * p.surveil_speed + radial_hat * (
            self.k_radial * (p.surveil_radius - r)
        )
        return desired.unit() * p.surveil_speed


@dataclass
class HoverController(Controller):
    """Command zero velocity. Used by the stop-first wait policy."""

    params: Params

    def command(self, t: float, evader: Evader, est: Estimate, dt: float) -> Vec2:
        return ZERO


@dataclass
class EscapeController(Controller):
    """Flee straight away from the estimated pursuer position.

    `v_flee=None` commands a velocity large enough to saturate the acceleration
    cap, which reproduces the certificate's assumption of "full acceleration,
    straight away, forever".  Setting a finite value introduces the speed cap
    the audit flagged as missing.
    """

    params: Params
    v_flee: Optional[float] = None
    tau_cmd: float = 0.2            # s, output smoother -- the controller's memory
    state: Vec2 = ZERO              # smoother state == last command sent

    def reset(self, evader: Evader, init: Optional[Vec2] = None) -> None:
        """Seed the memory. `init=None` means cold (zero)."""
        self.state = ZERO if init is None else init

    def raw_command(self, evader: Evader, est: Estimate) -> Vec2:
        away = (evader.pos - est.pos).unit()
        if away.norm == 0.0:
            # Exactly on top of the pursuer: keep going the way we already are.
            away = evader.vel.unit()
            if away.norm == 0.0:
                away = Vec2(1.0, 0.0)
        speed = self.v_flee if self.v_flee is not None else self._saturating_speed()
        return away * speed

    #: Stand-in for "no commanded speed limit at all".  It must be far beyond
    #: anything the drone can reach inside an evaluation window, and it must
    #: not depend on tau: if it did, it would act as an accidental speed cap
    #: that differs between tau values, and the tau comparison would be
    #: measuring that instead of the speed loop.
    UNBOUNDED_SPEED = 1.0e4

    def _saturating_speed(self) -> float:
        """Big enough that (v_cmd - v)/tau always exceeds a_max."""
        return self.UNBOUNDED_SPEED

    def command(self, t: float, evader: Evader, est: Estimate, dt: float) -> Vec2:
        raw = self.raw_command(evader, est)
        if self.tau_cmd <= 0.0:
            self.state = raw
        else:
            alpha = min(1.0, dt / self.tau_cmd)
            self.state = self.state + (raw - self.state) * alpha
        return self.state


@dataclass
class PrepareController(Controller):
    """A rough escape used during the wait: turn away, but only at survey speed.

    This is the handoff's `prepare` behaviour -- start escaping at once rather
    than keep surveying or brake to a hover.
    """

    params: Params

    def command(self, t: float, evader: Evader, est: Estimate, dt: float) -> Vec2:
        away = (evader.pos - est.pos).unit()
        return away * self.params.surveil_speed
