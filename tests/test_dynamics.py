import math
import unittest

from pursuit_evader.dynamics import Evader, Pursuer, closing_speed, retreat_speed
from pursuit_evader.params import Params
from pursuit_evader.vec import ZERO, Vec2


def settle(evader, v_cmd, duration, dt):
    steps = int(round(duration / dt))
    for _ in range(steps):
        evader.step(v_cmd, dt)
    return evader


class TestSpeedLoop(unittest.TestCase):
    """Step 3 of the handoff: dv/dt = (v_cmd - v)/tau, acceleration still capped."""

    def test_step_reaches_63_percent_after_tau(self):
        """The acceptance check named in the handoff."""
        for tau in (0.1, 0.3, 0.5):
            with self.subTest(tau=tau):
                p = Params(lag=True, tau=tau, dt=tau / 100.0)
                # Small step so the acceleration cap never binds.
                v_cmd = Vec2(0.5, 0.0)
                e = Evader(pos=ZERO, vel=ZERO, params=p)
                settle(e, v_cmd, tau, p.dt)
                fraction = e.vel.x / v_cmd.x
                self.assertAlmostEqual(fraction, 1.0 - math.exp(-1.0), delta=0.01)

    def test_step_settles_to_the_command(self):
        p = Params(lag=True, tau=0.3, dt=0.005)
        v_cmd = Vec2(0.5, 0.0)
        e = Evader(pos=ZERO, vel=ZERO, params=p)
        settle(e, v_cmd, 5.0 * p.tau, p.dt)
        self.assertAlmostEqual(e.vel.x, v_cmd.x, delta=0.01)

    def test_acceleration_cap_binds_on_a_large_step(self):
        p = Params(lag=True, tau=0.3, dt=0.005)
        e = Evader(pos=ZERO, vel=ZERO, params=p)
        a = e.step(Vec2(100.0, 0.0), p.dt)
        self.assertAlmostEqual(a.norm, p.a_max, delta=1e-9)

    def test_acceleration_never_exceeds_the_cap(self):
        p = Params(lag=True, tau=0.1, dt=0.002)
        e = Evader(pos=ZERO, vel=Vec2(0.0, 3.0), params=p)
        for _ in range(500):
            a = e.step(Vec2(-20.0, 7.0), p.dt)
            self.assertLessEqual(a.norm, p.a_max + 1e-9)

    def test_lag_off_reaches_the_command_in_one_step(self):
        """Lag off is the experiment 002 plant: instant, still acceleration capped."""
        p = Params(lag=False, dt=0.005)
        e = Evader(pos=ZERO, vel=ZERO, params=p)
        small = Vec2(0.01, 0.0)          # within a_max * dt = 0.02 m/s
        e.step(small, p.dt)
        self.assertAlmostEqual(e.vel.x, small.x, delta=1e-12)

    def test_lag_off_still_respects_the_acceleration_cap(self):
        p = Params(lag=False, dt=0.005)
        e = Evader(pos=ZERO, vel=ZERO, params=p)
        a = e.step(Vec2(50.0, 0.0), p.dt)
        self.assertAlmostEqual(a.norm, p.a_max, delta=1e-9)

    def test_smaller_tau_is_faster(self):
        cmd = Vec2(0.5, 0.0)
        speeds = []
        for tau in (0.1, 0.3, 0.5):
            p = Params(lag=True, tau=tau, dt=0.002)
            e = Evader(pos=ZERO, vel=ZERO, params=p)
            settle(e, cmd, 0.1, p.dt)
            speeds.append(e.speed)
        self.assertGreater(speeds[0], speeds[1])
        self.assertGreater(speeds[1], speeds[2])


class TestDoubleIntegrator(unittest.TestCase):
    def test_constant_acceleration_kinematics(self):
        """Saturated command for T seconds gives v = aT and x ~ aT^2/2."""
        p = Params(lag=False, dt=0.001)
        e = Evader(pos=ZERO, vel=ZERO, params=p)
        T = 2.0
        settle(e, Vec2(1000.0, 0.0), T, p.dt)
        self.assertAlmostEqual(e.vel.x, p.a_max * T, delta=0.01)
        self.assertAlmostEqual(e.pos.x, 0.5 * p.a_max * T * T, delta=0.05)

    def test_speed_cap_binds_when_set(self):
        p = Params(lag=False, dt=0.002, v_max=6.0)
        e = Evader(pos=ZERO, vel=ZERO, params=p)
        settle(e, Vec2(1000.0, 0.0), 10.0, p.dt)
        self.assertAlmostEqual(e.speed, 6.0, delta=1e-9)

    def test_uncapped_speed_keeps_growing(self):
        """The experiment 002 default: no speed cap at all."""
        p = Params(lag=False, dt=0.002, v_max=None)
        e = Evader(pos=ZERO, vel=ZERO, params=p)
        settle(e, Vec2(1000.0, 0.0), 10.0, p.dt)
        self.assertGreater(e.speed, 30.0)

    def test_realised_acceleration_is_reported_not_the_request(self):
        """When the speed cap bites, `accel` reports what actually happened."""
        p = Params(lag=False, dt=0.002, v_max=1.0)
        e = Evader(pos=ZERO, vel=Vec2(1.0, 0.0), params=p)
        a = e.step(Vec2(1000.0, 0.0), p.dt)
        self.assertAlmostEqual(a.norm, 0.0, delta=1e-9)


class TestPursuer(unittest.TestCase):
    def test_flies_at_its_speed(self):
        p = Params(dt=0.005)
        pur = Pursuer(pos=Vec2(-10.0, 0.0), params=p)
        pur.step(ZERO, p.dt)
        self.assertAlmostEqual(pur.vel.norm, p.v_pursuer, delta=1e-9)

    def test_heads_straight_at_the_target(self):
        p = Params(dt=0.005)
        pur = Pursuer(pos=Vec2(-10.0, 0.0), params=p)
        pur.step(ZERO, p.dt)
        self.assertAlmostEqual(pur.vel.y, 0.0, delta=1e-12)
        self.assertGreater(pur.vel.x, 0.0)

    def test_closes_on_a_stationary_target_at_v_pursuer(self):
        p = Params(dt=0.005)
        pur = Pursuer(pos=Vec2(-20.0, 0.0), params=p)
        for _ in range(int(1.0 / p.dt)):
            pur.step(ZERO, p.dt)
        self.assertAlmostEqual((pur.pos - ZERO).norm, 15.0, delta=0.01)


class TestRelativeSpeeds(unittest.TestCase):
    def test_retreat_speed_sign(self):
        # Evader at origin, pursuer to its left; fleeing right is positive v_r.
        v_r = retreat_speed(Vec2(2.0, 0.0), ZERO, Vec2(-5.0, 0.0))
        self.assertAlmostEqual(v_r, 2.0)
        v_r = retreat_speed(Vec2(-2.0, 0.0), ZERO, Vec2(-5.0, 0.0))
        self.assertAlmostEqual(v_r, -2.0)

    def test_retreat_speed_is_zero_when_moving_sideways(self):
        v_r = retreat_speed(Vec2(0.0, 3.0), ZERO, Vec2(-5.0, 0.0))
        self.assertAlmostEqual(v_r, 0.0)

    def test_closing_speed_when_evader_flees(self):
        """Pursuer gains at 5, evader runs away at 1 -> gap shrinks at 4 m/s."""
        p = Params()
        e = Evader(pos=ZERO, vel=Vec2(-1.0, 0.0), params=p)
        pur = Pursuer(pos=Vec2(10.0, 0.0), params=p, vel=Vec2(-5.0, 0.0))
        self.assertAlmostEqual(closing_speed(e, pur), 4.0)

    def test_closing_speed_when_evader_charges(self):
        """Head-on, both closing: 5 + 1 = 6 m/s."""
        p = Params()
        e = Evader(pos=ZERO, vel=Vec2(1.0, 0.0), params=p)
        pur = Pursuer(pos=Vec2(10.0, 0.0), params=p, vel=Vec2(-5.0, 0.0))
        self.assertAlmostEqual(closing_speed(e, pur), 6.0)

    def test_closing_speed_is_negative_when_outrunning(self):
        p = Params()
        e = Evader(pos=ZERO, vel=Vec2(-8.0, 0.0), params=p)
        pur = Pursuer(pos=Vec2(10.0, 0.0), params=p, vel=Vec2(-5.0, 0.0))
        self.assertAlmostEqual(closing_speed(e, pur), -3.0)


if __name__ == "__main__":
    unittest.main()
