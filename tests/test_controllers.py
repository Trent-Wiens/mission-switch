import math
import unittest

from pursuit_evader.certificate import Estimate
from pursuit_evader.controllers import (
    EscapeController,
    HoverController,
    PrepareController,
    SurveillanceController,
)
from pursuit_evader.dynamics import Evader
from pursuit_evader.params import Params
from pursuit_evader.vec import ZERO, Vec2, from_polar

FAR_AWAY = Estimate(pos=Vec2(-500.0, 0.0), vel=ZERO, age=0.0)


def fly(controller, evader, params, duration):
    steps = int(round(duration / params.dt))
    for i in range(steps):
        cmd = controller.command(i * params.dt, evader, FAR_AWAY, params.dt)
        evader.step(cmd, params.dt)
    return evader


class TestSurveillance(unittest.TestCase):
    def setUp(self):
        self.p = Params(lag=True, tau=0.3, dt=0.005)
        self.ctrl = SurveillanceController(self.p)

    def _on_circle(self):
        pos = from_polar(self.p.surveil_radius, 0.0)
        return Evader(pos=pos, vel=pos.unit().perp() * self.p.surveil_speed, params=self.p)

    def test_holds_the_circle(self):
        e = fly(self.ctrl, self._on_circle(), self.p, 60.0)
        self.assertAlmostEqual(e.pos.norm, self.p.surveil_radius, delta=0.5)

    def test_holds_the_survey_speed(self):
        e = fly(self.ctrl, self._on_circle(), self.p, 60.0)
        self.assertAlmostEqual(e.speed, self.p.surveil_speed, delta=0.1)

    def test_command_is_always_at_survey_speed(self):
        e = self._on_circle()
        cmd = self.ctrl.command(0.0, e, FAR_AWAY, self.p.dt)
        self.assertAlmostEqual(cmd.norm, self.p.surveil_speed, delta=1e-9)

    def test_converges_back_onto_the_circle_from_outside(self):
        pos = from_polar(45.0, 0.0)
        e = Evader(pos=pos, vel=pos.unit().perp() * self.p.surveil_speed, params=self.p)
        fly(self.ctrl, e, self.p, 60.0)
        self.assertAlmostEqual(e.pos.norm, self.p.surveil_radius, delta=1.0)

    def test_converges_back_onto_the_circle_from_inside(self):
        pos = from_polar(12.0, 0.0)
        e = Evader(pos=pos, vel=pos.unit().perp() * self.p.surveil_speed, params=self.p)
        fly(self.ctrl, e, self.p, 60.0)
        self.assertAlmostEqual(e.pos.norm, self.p.surveil_radius, delta=1.0)

    def test_stays_inside_the_geofence(self):
        e = self._on_circle()
        steps = int(round(120.0 / self.p.dt))
        for i in range(steps):
            cmd = self.ctrl.command(i * self.p.dt, e, FAR_AWAY, self.p.dt)
            e.step(cmd, self.p.dt)
            self.assertLess(e.pos.norm, self.p.geofence_radius)

    def test_goes_counter_clockwise_by_default(self):
        e = self._on_circle()
        start = e.pos.angle
        fly(self.ctrl, e, self.p, 5.0)
        self.assertGreater(e.pos.angle, start)

    def test_direction_can_be_reversed(self):
        ctrl = SurveillanceController(self.p, counter_clockwise=False)
        pos = from_polar(self.p.surveil_radius, 0.0)
        e = Evader(pos=pos, vel=-pos.unit().perp() * self.p.surveil_speed, params=self.p)
        fly(ctrl, e, self.p, 5.0)
        self.assertLess(e.pos.angle, 0.0)


class TestEscape(unittest.TestCase):
    def setUp(self):
        self.p = Params(lag=True, tau=0.3, dt=0.005)

    def test_raw_command_points_directly_away_from_the_pursuer(self):
        ctrl = EscapeController(self.p)
        e = Evader(pos=ZERO, vel=ZERO, params=self.p)
        est = Estimate(pos=Vec2(-10.0, 0.0), vel=ZERO, age=0.0)
        cmd = ctrl.raw_command(e, est)
        self.assertGreater(cmd.x, 0.0)
        self.assertAlmostEqual(cmd.y, 0.0, delta=1e-12)

    def test_default_command_saturates_the_acceleration_cap(self):
        """v_flee=None reproduces 'full acceleration, straight away'."""
        ctrl = EscapeController(self.p, tau_cmd=0.0)
        e = Evader(pos=ZERO, vel=ZERO, params=self.p)
        est = Estimate(pos=Vec2(-10.0, 0.0), vel=ZERO, age=0.0)
        ctrl.reset(e, e.vel)
        a = e.step(ctrl.command(0.0, e, est, self.p.dt), self.p.dt)
        self.assertAlmostEqual(a.norm, self.p.a_max, delta=1e-9)

    def test_finite_v_flee_caps_the_escape_speed(self):
        p = self.p.with_(v_max=6.0)
        ctrl = EscapeController(p, v_flee=6.0, tau_cmd=0.0)
        e = Evader(pos=ZERO, vel=ZERO, params=p)
        est = Estimate(pos=Vec2(-10.0, 0.0), vel=ZERO, age=0.0)
        ctrl.reset(e, e.vel)
        for i in range(int(10.0 / p.dt)):
            e.step(ctrl.command(i * p.dt, e, est, p.dt), p.dt)
        self.assertAlmostEqual(e.speed, 6.0, delta=0.05)

    def test_smoother_memory_starts_where_reset_puts_it(self):
        ctrl = EscapeController(self.p)
        e = Evader(pos=ZERO, vel=Vec2(0.0, 3.0), params=self.p)
        ctrl.reset(e, None)
        self.assertEqual(ctrl.state, ZERO)
        ctrl.reset(e, e.vel)
        self.assertEqual(ctrl.state, Vec2(0.0, 3.0))

    def test_smoother_moves_the_command_toward_the_raw_target(self):
        ctrl = EscapeController(self.p, v_flee=5.0, tau_cmd=0.2)
        e = Evader(pos=ZERO, vel=ZERO, params=self.p)
        est = Estimate(pos=Vec2(-10.0, 0.0), vel=ZERO, age=0.0)
        ctrl.reset(e, None)
        first = ctrl.command(0.0, e, est, self.p.dt)
        self.assertLess(first.norm, 5.0)
        for i in range(int(2.0 / self.p.dt)):
            cmd = ctrl.command(i * self.p.dt, e, est, self.p.dt)
        self.assertAlmostEqual(cmd.norm, 5.0, delta=0.05)


class TestWaitControllers(unittest.TestCase):
    def setUp(self):
        self.p = Params(lag=True, tau=0.3, dt=0.005)

    def test_hover_commands_zero(self):
        ctrl = HoverController(self.p)
        e = Evader(pos=ZERO, vel=Vec2(0.0, 3.0), params=self.p)
        self.assertEqual(ctrl.command(0.0, e, FAR_AWAY, self.p.dt), ZERO)

    def test_hover_brings_the_drone_to_rest(self):
        ctrl = HoverController(self.p)
        e = Evader(pos=ZERO, vel=Vec2(0.0, 3.0), params=self.p)
        fly(ctrl, e, self.p, 5.0)
        self.assertAlmostEqual(e.speed, 0.0, delta=0.05)

    def test_prepare_turns_away_at_survey_speed(self):
        ctrl = PrepareController(self.p)
        e = Evader(pos=ZERO, vel=Vec2(0.0, 3.0), params=self.p)
        est = Estimate(pos=Vec2(-10.0, 0.0), vel=ZERO, age=0.0)
        cmd = ctrl.command(0.0, e, est, self.p.dt)
        self.assertAlmostEqual(cmd.norm, self.p.surveil_speed, delta=1e-9)
        self.assertGreater(cmd.x, 0.0)


if __name__ == "__main__":
    unittest.main()
