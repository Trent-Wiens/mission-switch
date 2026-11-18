"""Plant models for the evader and the pursuer.

The evader is a double integrator driven by a *velocity* command.  Two modes:

    lag on   dv/dt = (v_cmd - v) / tau        (first-order speed loop)
    lag off  v     = v_cmd                    (instant, still acceleration capped)

In both modes the realised acceleration is capped at `a_max`, so "lag off" is
exactly the experiment 002 behaviour: ask to flee and you get full acceleration
on the same step.  Integration is semi-implicit Euler.
"""

from __future__ import annotations

from dataclasses import dataclass

from .params import Params
from .vec import ZERO, Vec2


@dataclass
class Evader:
    pos: Vec2
    vel: Vec2
    params: Params
    accel: Vec2 = ZERO          # realised acceleration on the last step
    cmd: Vec2 = ZERO            # velocity command on the last step

    def step(self, v_cmd: Vec2, dt: float) -> Vec2:
        """Advance one step under velocity command `v_cmd`.

        Returns the *realised* acceleration, which differs from the requested
        one whenever the acceleration cap or the speed cap binds.
        """
        p = self.params

        if p.lag:
            a_des = (v_cmd - self.vel) / p.tau
        else:
            # Close the whole error in one step, then let the cap bite.
            a_des = (v_cmd - self.vel) / dt

        a = a_des.clamp_norm(p.a_max)
        new_vel = self.vel + a * dt
        if p.v_max is not None:
            new_vel = new_vel.clamp_norm(p.v_max)

        # Report what actually happened, not what was asked for.
        self.accel = (new_vel - self.vel) / dt
        self.vel = new_vel
        self.pos = self.pos + self.vel * dt
        self.cmd = v_cmd
        return self.accel

    @property
    def speed(self) -> float:
        return self.vel.norm


@dataclass
class Pursuer:
    """Constant-speed pure pursuit: always heading straight at the evader.

    The turn is instantaneous, which is the worst case for the evader and the
    assumption the certificate is written against.
    """

    pos: Vec2
    params: Params
    vel: Vec2 = ZERO

    def step(self, target: Vec2, dt: float) -> None:
        heading = (target - self.pos).unit()
        self.vel = heading * self.params.v_pursuer
        self.pos = self.pos + self.vel * dt


def closing_speed(evader: Evader, pursuer: Pursuer) -> float:
    """Rate at which the gap is shrinking: -d(distance)/dt.

    Positive means the pursuer is gaining.
    """
    los = (pursuer.pos - evader.pos).unit()
    return (evader.vel - pursuer.vel).dot(los)


def retreat_speed(evader_vel: Vec2, evader_pos: Vec2, pursuer_pos: Vec2) -> float:
    """v_r: the evader's speed component directly away from the pursuer."""
    away = (evader_pos - pursuer_pos).unit()
    return evader_vel.dot(away)
