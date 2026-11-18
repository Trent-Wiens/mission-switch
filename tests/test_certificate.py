import unittest

from pursuit_evader.certificate import (
    DelayedEstimator,
    margin,
    margin_from_state,
    rho_allowance,
)
from pursuit_evader.params import Params
from pursuit_evader.vec import ZERO, Vec2


class TestMarginFormula(unittest.TestCase):
    """margin = d - rho - r_capture - max(0, v_p - v_r)^2 / (2 a_max)"""

    def setUp(self):
        self.p = Params()

    def test_matches_a_hand_computation(self):
        # d = 20, rho = 0, v_r = 1 -> closing 4, braking 16/8 = 2
        # margin = 20 - 0 - 3 - 2 = 15
        self.assertAlmostEqual(margin(20.0, 1.0, 0.0, self.p), 15.0)

    def test_stationary_evader_pays_the_full_braking_term(self):
        # v_r = 0 -> closing 5, braking 25/8 = 3.125
        self.assertAlmostEqual(margin(20.0, 0.0, 0.0, self.p), 20.0 - 3.0 - 3.125)

    def test_braking_term_vanishes_once_v_r_reaches_pursuer_speed(self):
        at_speed = margin(20.0, 5.0, 0.0, self.p)
        faster = margin(20.0, 9.0, 0.0, self.p)
        self.assertAlmostEqual(at_speed, 17.0)
        self.assertAlmostEqual(faster, 17.0)

    def test_retreating_backwards_costs_more(self):
        # v_r = -3 -> closing 8, braking 64/8 = 8
        self.assertAlmostEqual(margin(20.0, -3.0, 0.0, self.p), 20.0 - 3.0 - 8.0)

    def test_margin_decreases_with_rho(self):
        self.assertAlmostEqual(
            margin(20.0, 1.0, 2.0, self.p), margin(20.0, 1.0, 0.0, self.p) - 2.0
        )

    def test_extra_term_is_subtracted(self):
        """The slot where step 8's start-up cost lands."""
        base = margin(20.0, 1.0, 0.0, self.p)
        self.assertAlmostEqual(margin(20.0, 1.0, 0.0, self.p, extra=2.4), base - 2.4)

    def test_margin_is_negative_at_the_capture_radius(self):
        self.assertLess(margin(3.0, 0.0, 0.0, self.p), 0.0)


class TestMarginFromState(unittest.TestCase):
    def test_agrees_with_the_scalar_form(self):
        p = Params()
        evader_pos, pursuer_pos = Vec2(20.0, 0.0), ZERO
        evader_vel = Vec2(1.0, 0.0)          # straight away, v_r = 1
        self.assertAlmostEqual(
            margin_from_state(evader_pos, evader_vel, pursuer_pos, p),
            margin(20.0, 1.0, 0.0, p),
        )

    def test_sideways_motion_gives_zero_retreat_speed(self):
        p = Params()
        m = margin_from_state(Vec2(20.0, 0.0), Vec2(0.0, 3.0), ZERO, p)
        self.assertAlmostEqual(m, margin(20.0, 0.0, 0.0, p))


class TestRhoAllowance(unittest.TestCase):
    def test_zero_age_costs_nothing(self):
        self.assertAlmostEqual(rho_allowance(0.0, 3.0, Params()), 0.0)

    def test_grows_with_age_and_with_evader_speed(self):
        p = Params()
        # Gap can shrink at up to (v_pursuer + v_evader).
        self.assertAlmostEqual(rho_allowance(1.0, 3.0, p), 8.0)
        self.assertAlmostEqual(rho_allowance(2.0, 3.0, p), 16.0)
        self.assertGreater(rho_allowance(1.0, 5.0, p), rho_allowance(1.0, 3.0, p))


class TestDelayedEstimator(unittest.TestCase):
    def test_zero_delay_returns_the_current_state(self):
        est = DelayedEstimator(true_age=0.0, dt=0.005)
        est.update(Vec2(1.0, 0.0), ZERO)
        est.update(Vec2(2.0, 0.0), ZERO)
        self.assertAlmostEqual(est.estimate().pos.x, 2.0)
        self.assertAlmostEqual(est.estimate().age, 0.0)

    def test_delay_returns_an_older_state(self):
        dt = 0.005
        est = DelayedEstimator(true_age=0.05, dt=dt)       # 10 steps back
        for k in range(100):
            est.update(Vec2(float(k), 0.0), ZERO)
        e = est.estimate()
        self.assertAlmostEqual(e.pos.x, 89.0)
        self.assertAlmostEqual(e.age, 0.05, delta=1e-9)

    def test_assumed_age_defaults_to_the_true_age(self):
        est = DelayedEstimator(true_age=0.2, dt=0.005)
        self.assertAlmostEqual(est.assumed_age, 0.2)

    def test_assumed_age_can_understate_the_truth(self):
        """The experiment 002 false-certificate mechanism."""
        est = DelayedEstimator(true_age=1.0, assumed_age=0.0, dt=0.005)
        self.assertAlmostEqual(est.assumed_age, 0.0)
        self.assertAlmostEqual(est.true_age, 1.0)

    def test_raises_before_any_measurement(self):
        with self.assertRaises(RuntimeError):
            DelayedEstimator().estimate()

    def test_rejects_negative_age(self):
        with self.assertRaises(ValueError):
            DelayedEstimator(true_age=-1.0)


if __name__ == "__main__":
    unittest.main()
