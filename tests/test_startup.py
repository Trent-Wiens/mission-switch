import math
import unittest

from pursuit_evader.certificate import Estimate
from pursuit_evader.controllers import EscapeController
from pursuit_evader.dynamics import Evader
from pursuit_evader.params import Params
from pursuit_evader.startup import (
    BLEND_TIMES,
    METHODS,
    blend,
    cold,
    default_horizon,
    filtered_blend,
    predict_margin,
    safety_filter,
    warm,
)
from pursuit_evader.vec import ZERO, Vec2

PURSUER = Estimate(pos=Vec2(-15.0, 0.0), vel=Vec2(5.0, 0.0), age=0.0)


class TestMemoryInitialisation(unittest.TestCase):
    """Cold versus warm is entirely about where the controller's memory starts."""

    def setUp(self):
        self.p = Params(lag=True, tau=0.3, dt=0.005)
        self.evader = Evader(pos=ZERO, vel=Vec2(0.0, 3.0), params=self.p)
        self.escape = EscapeController(self.p, v_flee=5.0)

    def test_cold_start_memory_is_zero(self):
        cold().on_commit(self.evader, self.escape, self.evader.vel)
        self.assertEqual(self.escape.state, ZERO)

    def test_cold_start_first_command_does_not_match_the_motion(self):
        """This mismatch is the jolt the project is about."""
        cold().on_commit(self.evader, self.escape, self.evader.vel)
        first = self.escape.command(0.0, self.evader, PURSUER, self.p.dt)
        self.assertGreater((first - self.evader.vel).norm, 2.0)

    def test_warm_start_memory_is_the_measured_velocity(self):
        warm().on_commit(self.evader, self.escape, self.evader.vel)
        self.assertEqual(self.escape.state, self.evader.vel)

    def test_warm_start_first_command_matches_the_motion(self):
        warm().on_commit(self.evader, self.escape, self.evader.vel)
        first = self.escape.command(0.0, self.evader, PURSUER, self.p.dt)
        self.assertLess((first - self.evader.vel).norm, 0.2)

    def test_warm_beats_cold_on_the_first_command_mismatch(self):
        results = {}
        for name, make in (("cold", cold), ("warm", warm)):
            escape = EscapeController(self.p, v_flee=5.0)
            evader = Evader(pos=ZERO, vel=Vec2(0.0, 3.0), params=self.p)
            make().on_commit(evader, escape, evader.vel)
            cmd = escape.command(0.0, evader, PURSUER, self.p.dt)
            results[name] = (cmd - evader.vel).norm
        self.assertLess(results["warm"], results["cold"])


class TestBlend(unittest.TestCase):
    def setUp(self):
        self.p = Params(lag=True, tau=0.3, dt=0.005)
        self.evader = Evader(pos=ZERO, vel=Vec2(0.0, 3.0), params=self.p)
        self.outgoing = Vec2(0.0, 3.0)
        self.escape_cmd = Vec2(5.0, 0.0)

    def _cmd(self, method, t):
        return method.command(
            t, self.escape_cmd, self.outgoing, self.evader, PURSUER, self.p
        )

    def test_at_commit_the_blend_is_the_outgoing_command(self):
        m = blend(0.5)
        self.assertAlmostEqual((self._cmd(m, 0.0) - self.outgoing).norm, 0.0, delta=1e-12)

    def test_at_the_end_the_blend_is_the_escape_command(self):
        m = blend(0.5)
        self.assertAlmostEqual(
            (self._cmd(m, 0.5) - self.escape_cmd).norm, 0.0, delta=1e-12
        )

    def test_halfway_is_the_midpoint(self):
        m = blend(0.5)
        expected = (self.outgoing + self.escape_cmd) * 0.5
        self.assertAlmostEqual((self._cmd(m, 0.25) - expected).norm, 0.0, delta=1e-12)

    def test_blend_is_monotone_toward_the_escape_command(self):
        m = blend(1.0)
        gaps = [(self._cmd(m, t) - self.escape_cmd).norm for t in (0.0, 0.25, 0.5, 0.75)]
        self.assertEqual(gaps, sorted(gaps, reverse=True))

    def test_zero_blend_time_passes_the_escape_command_straight_through(self):
        m = cold()
        self.assertEqual(self._cmd(m, 0.0), self.escape_cmd)

    def test_all_requested_blend_times_build(self):
        for T_b in BLEND_TIMES:
            with self.subTest(T_b=T_b):
                self.assertAlmostEqual(blend(T_b).blend_time, T_b)


class TestPredictMargin(unittest.TestCase):
    def setUp(self):
        self.p = Params(lag=True, tau=0.3, dt=0.005)

    def test_fleeing_from_far_away_keeps_a_positive_margin(self):
        e = Evader(pos=ZERO, vel=Vec2(5.0, 0.0), params=self.p)
        est = Estimate(pos=Vec2(-50.0, 0.0), vel=Vec2(5.0, 0.0), age=0.0)
        self.assertGreater(predict_margin(e, est, Vec2(20.0, 0.0), self.p, 1.0), 0.0)

    def test_sitting_still_up_close_gives_a_negative_margin(self):
        e = Evader(pos=ZERO, vel=ZERO, params=self.p)
        est = Estimate(pos=Vec2(-4.0, 0.0), vel=Vec2(5.0, 0.0), age=0.0)
        self.assertLess(predict_margin(e, est, ZERO, self.p, 1.0), 0.0)

    def test_fleeing_predicts_a_better_margin_than_hovering(self):
        e = Evader(pos=ZERO, vel=ZERO, params=self.p)
        est = Estimate(pos=Vec2(-12.0, 0.0), vel=Vec2(5.0, 0.0), age=0.0)
        flee = predict_margin(e, est, Vec2(20.0, 0.0), self.p, 1.0)
        hover = predict_margin(e, est, ZERO, self.p, 1.0)
        self.assertGreater(flee, hover)


class TestSafetyFilter(unittest.TestCase):
    def setUp(self):
        self.p = Params(lag=True, tau=0.3, dt=0.005)

    def test_a_safe_command_passes_through_untouched(self):
        e = Evader(pos=ZERO, vel=Vec2(5.0, 0.0), params=self.p)
        est = Estimate(pos=Vec2(-60.0, 0.0), vel=Vec2(5.0, 0.0), age=0.0)
        candidate = Vec2(0.0, 3.0)
        out = safety_filter(candidate, Vec2(20.0, 0.0), e, est, self.p)
        self.assertEqual(out, candidate)

    def test_an_unsafe_command_is_moved_toward_the_fallback(self):
        e = Evader(pos=ZERO, vel=Vec2(0.0, 3.0), params=self.p)
        est = Estimate(pos=Vec2(-9.0, 0.0), vel=Vec2(5.0, 0.0), age=0.0)
        candidate = Vec2(0.0, 3.0)           # keep surveying: not safe here
        fallback = Vec2(20.0, 0.0)           # full flee
        out = safety_filter(candidate, fallback, e, est, self.p)
        self.assertGreater(out.x, candidate.x)

    def test_the_filter_changes_the_command_as_little_as_it_can(self):
        """The output should stop at the boundary, not jump to the fallback."""
        e = Evader(pos=ZERO, vel=Vec2(0.0, 3.0), params=self.p)
        est = Estimate(pos=Vec2(-9.0, 0.0), vel=Vec2(5.0, 0.0), age=0.0)
        candidate = Vec2(0.0, 3.0)
        fallback = Vec2(20.0, 0.0)
        out = safety_filter(candidate, fallback, e, est, self.p)
        self.assertLess((out - fallback).norm, (candidate - fallback).norm)
        self.assertGreater((out - fallback).norm, 0.0)

    def test_the_filtered_command_is_actually_safe(self):
        """Whatever it returns must pass the check that rejected the input."""
        e = Evader(pos=ZERO, vel=Vec2(0.0, 3.0), params=self.p)
        est = Estimate(pos=Vec2(-9.0, 0.0), vel=Vec2(5.0, 0.0), age=0.0)
        out = safety_filter(Vec2(0.0, 3.0), Vec2(20.0, 0.0), e, est, self.p)
        self.assertGreaterEqual(
            predict_margin(e, est, out, self.p, default_horizon(self.p)), 0.0
        )

    def test_hopeless_geometry_falls_back_rather_than_giving_up(self):
        e = Evader(pos=ZERO, vel=ZERO, params=self.p)
        est = Estimate(pos=Vec2(-3.2, 0.0), vel=Vec2(5.0, 0.0), age=0.0)
        fallback = Vec2(20.0, 0.0)
        self.assertEqual(safety_filter(ZERO, fallback, e, est, self.p), fallback)


class TestDefaultHorizon(unittest.TestCase):
    """The backup check must look far enough ahead to see the worst margin."""

    def test_covers_the_time_to_reach_the_pursuer_speed(self):
        p = Params(lag=True, tau=0.3, dt=0.005)
        self.assertGreaterEqual(
            default_horizon(p), (p.v_pursuer + p.surveil_speed) / p.a_max
        )

    def test_grows_with_tau_once_the_lag_dominates(self):
        slow = Params(lag=True, tau=1.5, dt=0.005)
        self.assertAlmostEqual(default_horizon(slow), 6.0)

    def test_a_short_horizon_can_miss_an_unsafe_command(self):
        """Why the default is not simply 4*tau: at 1.2 s this case looks safe."""
        p = Params(lag=True, tau=0.3, dt=0.005)
        e = Evader(pos=ZERO, vel=Vec2(0.0, 3.0), params=p)
        est = Estimate(pos=Vec2(-11.0, 0.0), vel=Vec2(5.0, 0.0), age=0.0)
        survey = Vec2(0.0, 3.0)
        self.assertGreater(predict_margin(e, est, survey, p, 1.2), 0.0)
        self.assertLess(predict_margin(e, est, survey, p, default_horizon(p)), 0.0)


class TestMethodRegistry(unittest.TestCase):
    def test_all_four_methods_build_and_are_named(self):
        self.assertEqual(set(METHODS), {"cold", "warm", "blend", "filtered"})
        for key, make in METHODS.items():
            with self.subTest(method=key):
                self.assertTrue(make().name)

    def test_blend_and_filtered_keep_cold_memory_by_default(self):
        """So the blend's effect is isolated from the warm-start effect."""
        self.assertFalse(blend().warm_memory)
        self.assertFalse(filtered_blend().warm_memory)

    def test_only_filtered_runs_the_filter(self):
        self.assertFalse(cold().filtered)
        self.assertFalse(warm().filtered)
        self.assertFalse(blend().filtered)
        self.assertTrue(filtered_blend().filtered)


if __name__ == "__main__":
    unittest.main()
