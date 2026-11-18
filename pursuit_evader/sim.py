"""The simulation loop.

Timeline.  t = 0 is *detection*.  The escape controller may not take charge
until t = h (the *commit*).  Every run is scored over the same window,
[0, t_eval], which closes the audit's second caveat that policies had been
compared over different windows.

    0 ---- wait policy (defer | stop_first | prepare | forecast_forward) ---- h
    h ---- escape controller, brought online by a StartupMethod ---------- t_eval
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional

from .certificate import DelayedEstimator, Estimate, margin, rho_allowance
from .controllers import (
    Controller,
    EscapeController,
    HoverController,
    PrepareController,
    SurveillanceController,
)
from .dynamics import Evader, Pursuer
from .params import DEFAULT, Params
from .startup import StartupMethod, cold
from .vec import ZERO, Vec2, from_polar

WAIT_POLICIES = ("defer", "stop_first", "prepare", "forecast_forward")


class Sample(NamedTuple):
    t: float
    evader_pos: Vec2
    evader_vel: Vec2
    evader_accel: Vec2
    cmd: Vec2
    pursuer_pos: Vec2
    distance: float             # true distance
    margin: float               # as the controller computes it (stale estimate)
    true_margin: float          # with a perfect, current estimate
    cert_holds: bool
    committed: bool


@dataclass
class Scenario:
    """One detection case."""

    detect_range: float                 # m, true distance at detection
    bearing: float                      # rad, pursuer direction relative to evader heading
    h: float = 1.0                      # s, wait before commit
    policy: str = "prepare"
    startup: StartupMethod = field(default_factory=cold)
    params: Params = DEFAULT
    theta0: float = 0.0                 # evader's starting angle on its circle
    true_age: float = 0.0               # s, true staleness of the pursuer estimate
    assumed_age: Optional[float] = None  # s, what rho budgets for

    # Escape controller settings.
    #
    # v_flee=None commands a velocity large enough to saturate the acceleration
    # cap.  That reproduces experiment 002 -- but it also makes the speed loop
    # invisible, because (v_cmd - v)/tau is clipped to a_max at every tau, and
    # it makes the commanded-acceleration metric meaningless.  Set a finite
    # value for any run where the lag is supposed to matter.
    v_flee: Optional[float] = None
    tau_cmd: float = 0.2                # s, the escape controller's own memory

    def __post_init__(self) -> None:
        if self.policy not in WAIT_POLICIES:
            raise ValueError(f"unknown policy {self.policy!r}; expected one of {WAIT_POLICIES}")
        if self.h < 0.0:
            raise ValueError("h must be non-negative")


@dataclass
class Result:
    scenario: Scenario
    trace: List[Sample]

    captured: bool = False
    t_capture: Optional[float] = None
    captured_after_commit: bool = False

    cert_held_at_commit: bool = False
    false_certificate: bool = False

    min_margin_after_commit: float = float("inf")
    min_distance: float = float("inf")

    # The jolt, as the section 4 table defines it.
    #
    # `cmd_vel_jump` is the discontinuity in the velocity command at commit.
    # `peak_cmd_accel` is the acceleration the speed loop is *asked* for,
    # (v_cmd - v)/tau before the cap is applied -- not d(v_cmd)/dt, which is
    # unbounded at a step and therefore only a function of the step size.
    cmd_vel_jump: float = 0.0           # m/s, |v_cmd(commit+) - v_cmd(commit-)|
    cmd_accel_jump: float = 0.0         # m/s^2, change in demanded accel at commit
    peak_cmd_accel: float = 0.0         # m/s^2, max demanded |(v_cmd - v)/tau|
    peak_accel: float = 0.0             # m/s^2, max realised |a| (<= a_max)

    # Slew rate of the command once the new controller is running, measured
    # strictly after the commit step so the handover discontinuity (which is
    # `cmd_vel_jump`, and is step-size dependent) does not contaminate it.
    # This is what separates one blend length from another.
    peak_cmd_slew: float = 0.0          # m/s^2, max |d v_cmd / dt| after commit

    geofence_breach: bool = False
    max_radius: float = 0.0

    @property
    def survived(self) -> bool:
        return not self.captured


def _wait_controller(policy: str, params: Params) -> Controller:
    if policy == "defer":
        return SurveillanceController(params)
    if policy == "stop_first":
        return HoverController(params)
    if policy in ("prepare", "forecast_forward"):
        return PrepareController(params)
    raise ValueError(policy)


def _forecast_safe_at_commit(
    evader: Evader,
    est: Estimate,
    t_to_commit: float,
    surveil: SurveillanceController,
    params: Params,
    substeps: int = 20,
) -> bool:
    """Would the certificate still hold at commit if we kept surveying?

    This is the handoff's `forecast_forward` check: keep surveying only when it
    can be proved that the drone is still safe at the moment of the switch.
    """
    if t_to_commit <= 0.0:
        return True

    h = t_to_commit / substeps
    probe = Evader(pos=evader.pos, vel=evader.vel, params=params)
    p_pos = est.pos

    for _ in range(substeps):
        v_cmd = surveil.command(0.0, probe, est, h)
        probe.step(v_cmd, h)
        p_pos = p_pos + (probe.pos - p_pos).unit() * params.v_pursuer * h

    rel = probe.pos - p_pos
    v_r = probe.vel.dot(rel.unit())
    rho = rho_allowance(est.age, probe.speed, params)
    return margin(rel.norm, v_r, rho, params) >= 0.0


def run(scenario: Scenario) -> Result:
    """Run one detection case over the fixed evaluation window."""
    p = scenario.params
    dt = p.dt

    # --- initial geometry, defined at the moment of detection ---------------
    center = ZERO
    evader_pos = from_polar(p.surveil_radius, scenario.theta0)
    heading = evader_pos.unit().perp()                   # counter-clockwise
    evader_vel = heading * p.surveil_speed
    pursuer_pos = evader_pos + from_polar(
        scenario.detect_range, heading.angle + scenario.bearing
    )

    evader = Evader(pos=evader_pos, vel=evader_vel, params=p)
    pursuer = Pursuer(pos=pursuer_pos, params=p)
    pursuer.vel = (evader_pos - pursuer_pos).unit() * p.v_pursuer

    estimator = DelayedEstimator(
        true_age=scenario.true_age, assumed_age=scenario.assumed_age, dt=dt
    )
    # Prime the buffer by back-propagating the pursuer along its heading, so the
    # estimate is already `true_age` seconds old at detection.
    for k in range(estimator._depth - 1, -1, -1):
        estimator.update(pursuer.pos - pursuer.vel * (k * dt), pursuer.vel)

    surveil = SurveillanceController(p, center=center)
    waiter = _wait_controller(scenario.policy, p)
    escape = EscapeController(p, v_flee=scenario.v_flee, tau_cmd=scenario.tau_cmd)
    startup = scenario.startup

    result = Result(scenario=scenario, trace=[])
    committed = False
    last_cmd = evader_vel                 # outgoing controller's command
    prev_cmd: Optional[Vec2] = None
    t_commit = scenario.h

    n_steps = int(round(p.t_eval / dt))
    for i in range(n_steps + 1):
        t = i * dt
        estimator.update(pursuer.pos, pursuer.vel)
        est = estimator.estimate()

        # --- certificate, as the controller sees it -------------------------
        rel_est = evader.pos - est.pos
        v_r_est = evader.vel.dot(rel_est.unit())
        rho = rho_allowance(estimator.assumed_age, evader.speed, p)
        m = margin(rel_est.norm, v_r_est, rho, p)

        rel_true = evader.pos - pursuer.pos
        true_dist = rel_true.norm
        true_m = margin(
            true_dist, evader.vel.dot(rel_true.unit()), 0.0, p
        )

        # --- commit ---------------------------------------------------------
        just_committed = False
        if not committed and t >= t_commit - 1e-12:
            startup.on_commit(evader, escape, last_cmd)
            committed = True
            just_committed = True
            result.cert_held_at_commit = m >= 0.0

        # --- pick a command --------------------------------------------------
        if committed:
            escape_cmd = escape.command(t, evader, est, dt)
            outgoing = surveil.command(t, evader, est, dt)
            v_cmd = startup.command(
                t - t_commit, escape_cmd, outgoing, evader, est, p
            )
        else:
            if scenario.policy == "forecast_forward" and _forecast_safe_at_commit(
                evader, est, t_commit - t, surveil, p
            ):
                v_cmd = surveil.command(t, evader, est, dt)
            else:
                v_cmd = waiter.command(t, evader, est, dt)

        # --- metrics ---------------------------------------------------------
        # Acceleration demanded of the speed loop, before the cap bites.
        demanded = (v_cmd - evader.vel) / p.tau
        if just_committed and prev_cmd is not None:
            result.cmd_vel_jump = (v_cmd - prev_cmd).norm
            prev_demanded = (prev_cmd - evader.vel) / p.tau
            result.cmd_accel_jump = (demanded - prev_demanded).norm
        if committed:
            result.peak_cmd_accel = max(result.peak_cmd_accel, demanded.norm)
            result.min_margin_after_commit = min(result.min_margin_after_commit, m)
            result.peak_accel = max(result.peak_accel, evader.accel.norm)
            if not just_committed and prev_cmd is not None:
                result.peak_cmd_slew = max(
                    result.peak_cmd_slew, (v_cmd - prev_cmd).norm / dt
                )

        result.min_distance = min(result.min_distance, true_dist)
        radius = evader.pos.norm
        result.max_radius = max(result.max_radius, radius)
        if radius > p.geofence_radius:
            result.geofence_breach = True

        result.trace.append(
            Sample(
                t=t,
                evader_pos=evader.pos,
                evader_vel=evader.vel,
                evader_accel=evader.accel,
                cmd=v_cmd,
                pursuer_pos=pursuer.pos,
                distance=true_dist,
                margin=m,
                true_margin=true_m,
                cert_holds=m >= 0.0,
                committed=committed,
            )
        )

        # --- capture check ----------------------------------------------------
        if true_dist <= p.capture_radius:
            result.captured = True
            result.t_capture = t
            result.captured_after_commit = committed
            break

        # --- advance ----------------------------------------------------------
        evader.step(v_cmd, dt)
        pursuer.step(evader.pos, dt)
        if not committed:
            last_cmd = v_cmd
        prev_cmd = v_cmd

    result.false_certificate = (
        result.cert_held_at_commit and result.captured_after_commit
    )
    if result.min_margin_after_commit == float("inf"):
        result.min_margin_after_commit = float("nan")
    return result
