import math
import unittest

from pursuit_evader.params import Params
from pursuit_evader.scenarios import BROADSIDE, FROM_BEHIND, HEAD_ON, basic_set
from pursuit_evader.sim import WAIT_POLICIES, Scenario, run
from pursuit_evader.startup import blend, cold, filtered_blend, warm

FAST = Params(lag=True, tau=0.3, dt=0.01, t_eval=20.0)


class TestScenarioValidation(unittest.TestCase):
    def test_rejects_an_unknown_policy(self):
        with self.assertRaises(ValueError):
            Scenario(detect_range=20.0, bearing=0.0, policy="teleport")

    def test_rejects_a_negative_wait(self):
        with self.assertRaises(ValueError):
            Scenario(detect_range=20.0, bearing=0.0, h=-1.0)

    def test_every_named_policy_runs(self):
        for policy in WAIT_POLICIES:
            with self.subTest(policy=policy):
                r = run(Scenario(detect_range=25.0, bearing=HEAD_ON,
                                 policy=policy, params=FAST))
                self.assertGreater(len(r.trace), 1)


class TestGeometry(unittest.TestCase):
    def test_detection_range_is_the_true_initial_distance(self):
        for bearing in (HEAD_ON, BROADSIDE, FROM_BEHIND):
            for rng in (8.0, 20.0, 35.0):
                with self.subTest(bearing=bearing, range=rng):
                    r = run(Scenario(detect_range=rng, bearing=bearing, params=FAST))
                    self.assertAlmostEqual(r.trace[0].distance, rng, delta=1e-6)

    def test_evader_starts_on_its_surveillance_circle_at_survey_speed(self):
        r = run(Scenario(detect_range=20.0, bearing=HEAD_ON, params=FAST))
        first = r.trace[0]
        self.assertAlmostEqual(first.evader_pos.norm, FAST.surveil_radius, delta=1e-9)
        self.assertAlmostEqual(first.evader_vel.norm, FAST.surveil_speed, delta=1e-9)

    def test_head_on_puts_the_pursuer_ahead_of_the_evader(self):
        r = run(Scenario(detect_range=20.0, bearing=HEAD_ON, params=FAST))
        first = r.trace[0]
        ahead = (first.pursuer_pos - first.evader_pos).unit()
        self.assertGreater(ahead.dot(first.evader_vel.unit()), 0.99)

    def test_from_behind_puts_the_pursuer_astern(self):
        r = run(Scenario(detect_range=20.0, bearing=FROM_BEHIND, params=FAST))
        first = r.trace[0]
        astern = (first.pursuer_pos - first.evader_pos).unit()
        self.assertLess(astern.dot(first.evader_vel.unit()), -0.99)


class TestOutcomes(unittest.TestCase):
    def test_a_distant_pursuer_never_catches_the_drone(self):
        r = run(Scenario(detect_range=50.0, bearing=HEAD_ON,
                         policy="prepare", params=FAST))
        self.assertFalse(r.captured)
        self.assertIsNone(r.t_capture)

    def test_a_pursuer_already_inside_the_capture_radius_wins_immediately(self):
        r = run(Scenario(detect_range=2.0, bearing=HEAD_ON, params=FAST))
        self.assertTrue(r.captured)
        self.assertAlmostEqual(r.t_capture, 0.0, delta=1e-9)

    def test_capture_is_declared_at_the_capture_radius(self):
        r = run(Scenario(detect_range=4.0, bearing=HEAD_ON,
                         policy="stop_first", h=2.0, params=FAST))
        self.assertTrue(r.captured)
        self.assertLessEqual(r.min_distance, FAST.capture_radius + 1e-9)

    def test_the_run_stops_at_capture(self):
        r = run(Scenario(detect_range=5.0, bearing=HEAD_ON,
                         policy="stop_first", h=2.0, params=FAST))
        self.assertTrue(r.captured)
        self.assertAlmostEqual(r.trace[-1].t, r.t_capture, delta=1e-9)

    def test_survived_is_the_complement_of_captured(self):
        r = run(Scenario(detect_range=50.0, bearing=HEAD_ON, params=FAST))
        self.assertEqual(r.survived, not r.captured)


class TestFixedEvaluationWindow(unittest.TestCase):
    """Closes the audit's second caveat: one window for every policy."""

    def test_all_policies_are_scored_over_the_same_window(self):
        durations = []
        for policy in WAIT_POLICIES:
            r = run(Scenario(detect_range=50.0, bearing=HEAD_ON,
                             policy=policy, params=FAST))
            self.assertFalse(r.captured, f"{policy} should survive at 50 m")
            durations.append(r.trace[-1].t)
        for d in durations:
            self.assertAlmostEqual(d, FAST.t_eval, delta=FAST.dt)

    def test_the_window_length_is_configurable(self):
        p = FAST.with_(t_eval=8.0)
        r = run(Scenario(detect_range=50.0, bearing=HEAD_ON, params=p))
        self.assertAlmostEqual(r.trace[-1].t, 8.0, delta=p.dt)


class TestCommit(unittest.TestCase):
    def test_commit_happens_at_t_equals_h(self):
        for h in (0.5, 1.0, 2.0):
            with self.subTest(h=h):
                r = run(Scenario(detect_range=40.0, bearing=HEAD_ON,
                                 h=h, params=FAST))
                first_committed = next(s.t for s in r.trace if s.committed)
                self.assertAlmostEqual(first_committed, h, delta=FAST.dt)

    def test_nothing_is_committed_before_h(self):
        r = run(Scenario(detect_range=40.0, bearing=HEAD_ON, h=1.0, params=FAST))
        for s in r.trace:
            if s.t < 1.0 - FAST.dt:
                self.assertFalse(s.committed)

    def test_h_of_zero_commits_at_once(self):
        r = run(Scenario(detect_range=40.0, bearing=HEAD_ON, h=0.0, params=FAST))
        self.assertTrue(r.trace[0].committed)


class TestJoltMetrics(unittest.TestCase):
    def test_cold_start_jolts_more_than_warm_start(self):
        """The headline comparison the results table is built around."""
        common = dict(detect_range=18.0, bearing=HEAD_ON, h=1.0,
                      policy="defer", params=FAST)
        cold_r = run(Scenario(startup=cold(), **common))
        warm_r = run(Scenario(startup=warm(), **common))
        self.assertGreater(cold_r.cmd_vel_jump, warm_r.cmd_vel_jump)

    def test_a_blend_reduces_the_commanded_acceleration_jolt(self):
        common = dict(detect_range=18.0, bearing=HEAD_ON, h=1.0,
                      policy="defer", params=FAST)
        cold_r = run(Scenario(startup=cold(), **common))
        blend_r = run(Scenario(startup=blend(1.0), **common))
        self.assertLess(blend_r.peak_cmd_accel, cold_r.peak_cmd_accel)

    def test_longer_blends_slew_the_command_more_slowly(self):
        common = dict(detect_range=25.0, bearing=HEAD_ON, h=1.0, policy="defer",
                      params=FAST.with_(v_max=8.0), v_flee=8.0)
        slews = [
            run(Scenario(startup=blend(T_b), **common)).peak_cmd_slew
            for T_b in (0.25, 0.5, 1.0, 2.0)
        ]
        self.assertEqual(slews, sorted(slews, reverse=True))

    def test_doubling_the_blend_time_roughly_halves_the_slew(self):
        """The command travels a fixed distance over T_b, so slew ~ 1/T_b."""
        common = dict(detect_range=25.0, bearing=HEAD_ON, h=1.0, policy="defer",
                      params=FAST.with_(v_max=8.0), v_flee=8.0)
        half = run(Scenario(startup=blend(1.0), **common)).peak_cmd_slew
        full = run(Scenario(startup=blend(2.0), **common)).peak_cmd_slew
        self.assertAlmostEqual(half / full, 2.0, delta=0.2)

    def test_the_jolt_ordering_is_cold_worst_blend_best(self):
        """The headline result: cold > warm > blend on both jolt measures."""
        common = dict(detect_range=25.0, bearing=HEAD_ON, h=1.0, policy="defer",
                      params=FAST.with_(v_max=8.0), v_flee=8.0)
        c = run(Scenario(startup=cold(), **common))
        w = run(Scenario(startup=warm(), **common))
        b = run(Scenario(startup=blend(0.5), **common))

        self.assertGreater(c.cmd_vel_jump, w.cmd_vel_jump)
        self.assertGreater(w.cmd_vel_jump, b.cmd_vel_jump)
        self.assertGreater(c.cmd_accel_jump, w.cmd_accel_jump)
        self.assertGreater(w.cmd_accel_jump, b.cmd_accel_jump)

    def test_a_blend_is_continuous_at_commit(self):
        """A blend starts from the outgoing command, so there is no step.

        The residual is one step of the surveillance command's own motion,
        O(dt), not a handover discontinuity.
        """
        common = dict(detect_range=25.0, bearing=HEAD_ON, h=1.0, policy="defer",
                      params=FAST.with_(v_max=8.0), v_flee=8.0)
        b = run(Scenario(startup=blend(0.5), **common))
        c = run(Scenario(startup=cold(), **common))
        self.assertLess(b.cmd_vel_jump, 0.01)
        self.assertLess(b.cmd_vel_jump, c.cmd_vel_jump / 100.0)

    def test_the_commit_residual_of_a_blend_shrinks_with_the_step_size(self):
        """Confirms that residual is O(dt) and not a real discontinuity."""
        jumps = []
        for dt in (0.01, 0.005, 0.0025):
            r = run(Scenario(detect_range=25.0, bearing=HEAD_ON, h=1.0,
                             policy="defer", startup=blend(0.5),
                             params=FAST.with_(dt=dt, v_max=8.0), v_flee=8.0))
            jumps.append(r.cmd_vel_jump)
        self.assertEqual(jumps, sorted(jumps, reverse=True))

    def test_the_demanded_acceleration_jump_converges_as_the_step_shrinks(self):
        """Why the metric is (v_cmd - v)/tau and not d(v_cmd)/dt.

        d(v_cmd)/dt at a step is just dv/dt and diverges as dt -> 0. The
        demanded acceleration converges instead; the residual spread is the
        escape controller's own first-order smoother, which advances by
        dt/tau_cmd on its first step.
        """
        jumps = []
        for dt in (0.01, 0.005, 0.0025):
            r = run(Scenario(detect_range=25.0, bearing=HEAD_ON, h=1.0,
                             policy="defer", startup=cold(),
                             params=FAST.with_(dt=dt, v_max=8.0), v_flee=8.0))
            jumps.append(r.cmd_accel_jump)
        self.assertGreater(abs(jumps[0] - jumps[1]), abs(jumps[1] - jumps[2]))
        self.assertLess(abs(jumps[1] - jumps[2]) / jumps[-1], 0.1)

    def test_realised_acceleration_never_exceeds_the_cap_even_on_a_cold_start(self):
        r = run(Scenario(detect_range=25.0, bearing=HEAD_ON, h=1.0, policy="defer",
                         startup=cold(), params=FAST.with_(v_max=8.0), v_flee=8.0))
        self.assertLessEqual(r.peak_accel, FAST.a_max + 1e-6)

    def test_metrics_are_recorded_only_after_commit(self):
        r = run(Scenario(detect_range=40.0, bearing=HEAD_ON, h=1.0,
                         policy="defer", params=FAST))
        self.assertGreater(r.peak_cmd_accel, 0.0)
        self.assertTrue(math.isfinite(r.min_margin_after_commit))


class TestCertificateReporting(unittest.TestCase):
    def test_a_far_pursuer_certifies_at_commit(self):
        r = run(Scenario(detect_range=50.0, bearing=HEAD_ON, params=FAST))
        self.assertTrue(r.cert_held_at_commit)
        self.assertFalse(r.false_certificate)

    def test_a_stale_estimate_can_produce_a_false_certificate(self):
        """Experiment 002: an estimate 1 s staler than assumed certifies and loses."""
        found = False
        p = FAST.with_(t_eval=15.0)
        for rng in [6.0 + 0.5 * k for k in range(40)]:
            r = run(Scenario(detect_range=rng, bearing=HEAD_ON, h=1.0,
                             policy="defer", startup=cold(),
                             true_age=1.0, assumed_age=0.0, params=p))
            if r.false_certificate:
                found = True
                break
        self.assertTrue(found, "no false certificate found with a 1 s stale estimate")

    def test_an_honest_estimate_does_not_produce_false_certificates(self):
        """With rho budgeting for the true age, the certificate should hold up."""
        p = FAST.with_(t_eval=15.0)
        for rng in [6.0 + 1.0 * k for k in range(20)]:
            r = run(Scenario(detect_range=rng, bearing=HEAD_ON, h=1.0,
                             policy="prepare", startup=warm(),
                             true_age=1.0, assumed_age=1.0, params=p))
            self.assertFalse(
                r.false_certificate, f"false certificate at range {rng}"
            )


class TestGeofence(unittest.TestCase):
    def test_the_drone_stays_inside_the_geofence_before_commit(self):
        """Only the wait phase: after commit the escape runs out of the fence."""
        r = run(Scenario(detect_range=60.0, bearing=HEAD_ON,
                         policy="defer", h=2.0, params=FAST))
        before = [s for s in r.trace if not s.committed]
        self.assertGreater(len(before), 1)
        self.assertLess(max(s.evader_pos.norm for s in before),
                        FAST.geofence_radius)

    def test_an_uncapped_escape_breaches_the_geofence_and_it_is_recorded(self):
        """With no speed cap the drone runs clean out of the fence. Noted, not fixed:
        keeping the escape inside the geofence is not in this project's scope."""
        p = FAST.with_(t_eval=30.0, v_max=None)
        r = run(Scenario(detect_range=12.0, bearing=FROM_BEHIND,
                         policy="prepare", h=0.5, params=p))
        self.assertGreater(r.max_radius, p.geofence_radius)
        self.assertTrue(r.geofence_breach)


class TestTraceIntegrity(unittest.TestCase):
    def test_time_is_uniform_and_increasing(self):
        r = run(Scenario(detect_range=50.0, bearing=HEAD_ON, params=FAST))
        for a, b in zip(r.trace, r.trace[1:]):
            self.assertAlmostEqual(b.t - a.t, FAST.dt, delta=1e-9)

    def test_recorded_distance_matches_the_recorded_positions(self):
        r = run(Scenario(detect_range=25.0, bearing=BROADSIDE, params=FAST))
        for s in r.trace[::50]:
            self.assertAlmostEqual(
                s.distance, (s.evader_pos - s.pursuer_pos).norm, delta=1e-9
            )

    def test_cert_holds_agrees_with_the_margin_sign(self):
        r = run(Scenario(detect_range=12.0, bearing=HEAD_ON, params=FAST))
        for s in r.trace[::25]:
            self.assertEqual(s.cert_holds, s.margin >= 0.0)

    def test_acceleration_in_the_trace_never_exceeds_the_cap(self):
        r = run(Scenario(detect_range=15.0, bearing=HEAD_ON,
                         startup=cold(), params=FAST))
        for s in r.trace:
            self.assertLessEqual(s.evader_accel.norm, FAST.a_max + 1e-6)

    def test_pursuer_speed_in_the_trace_is_constant(self):
        r = run(Scenario(detect_range=25.0, bearing=BROADSIDE, params=FAST))
        for a, b in zip(r.trace[1:], r.trace[2:]):
            step = (b.pursuer_pos - a.pursuer_pos).norm / FAST.dt
            self.assertAlmostEqual(step, FAST.v_pursuer, delta=1e-6)


class TestFleeSpeedMatters(unittest.TestCase):
    """v_flee=None makes the speed loop invisible. Pinned so it is not forgotten."""

    def test_a_saturating_command_gives_full_authority_at_every_tau(self):
        """The reason v_flee=None cannot be used to study the speed loop.

        The command is so far from the current velocity that (v_cmd - v)/tau
        clips to a_max on every step whatever tau is, so the escape gets full
        acceleration regardless -- exactly the assumption the project exists to
        remove. (The lag still degrades circle-tracking *before* commit, so the
        trajectories differ; it is the escape authority that is unaffected.)
        """
        for tau in (0.1, 0.3, 0.5):
            with self.subTest(tau=tau):
                r = run(Scenario(detect_range=20.0, bearing=HEAD_ON, h=1.0,
                                 policy="defer", startup=warm(), v_flee=None,
                                 params=FAST.with_(tau=tau, v_max=None)))
                post = [s for s in r.trace if s.committed][1:]
                self.assertAlmostEqual(
                    min(s.evader_accel.norm for s in post), FAST.a_max, delta=1e-6
                )

    def test_a_finite_flee_speed_lets_the_acceleration_fall_below_the_cap(self):
        r = run(Scenario(detect_range=20.0, bearing=HEAD_ON, h=1.0,
                         policy="defer", startup=warm(), v_flee=8.0,
                         params=FAST.with_(tau=0.3, v_max=8.0)))
        post = [s for s in r.trace if s.committed][1:]
        self.assertLess(min(s.evader_accel.norm for s in post), FAST.a_max - 0.5)

    def test_a_finite_flee_speed_lets_tau_change_the_outcome(self):
        traces = []
        for tau in (0.1, 0.5):
            r = run(Scenario(detect_range=20.0, bearing=HEAD_ON, h=1.0,
                             policy="defer", startup=warm(), v_flee=8.0,
                             params=FAST.with_(tau=tau, v_max=8.0)))
            traces.append(r.trace[-1].evader_pos)
        self.assertGreater((traces[0] - traces[1]).norm, 0.5)


class TestLagSwitch(unittest.TestCase):
    def test_lag_can_be_turned_off(self):
        no_lag = FAST.with_(lag=False)
        r = run(Scenario(detect_range=20.0, bearing=HEAD_ON, params=no_lag))
        self.assertGreater(len(r.trace), 1)

    def test_lag_off_tracks_the_command_more_closely(self):
        common = dict(detect_range=20.0, bearing=HEAD_ON, h=1.0,
                      policy="defer", startup=warm())
        lagged = run(Scenario(params=FAST.with_(tau=0.5), **common))
        prompt = run(Scenario(params=FAST.with_(lag=False), **common))

        def tracking_error(result):
            window = [s for s in result.trace if 1.0 <= s.t <= 2.0]
            return max((s.evader_vel - s.cmd).norm for s in window)

        self.assertLess(tracking_error(prompt), tracking_error(lagged))

    def test_a_longer_tau_tracks_worse(self):
        common = dict(detect_range=20.0, bearing=HEAD_ON, h=1.0,
                      policy="defer", startup=warm())

        def tracking_error(tau):
            r = run(Scenario(params=FAST.with_(tau=tau), **common))
            window = [s for s in r.trace if 1.0 <= s.t <= 2.0]
            return max((s.evader_vel - s.cmd).norm for s in window)

        self.assertLess(tracking_error(0.1), tracking_error(0.5))

    def test_every_requested_tau_runs(self):
        for tau in (0.1, 0.3, 0.5):
            with self.subTest(tau=tau):
                p = FAST.with_(tau=tau, dt=min(0.01, tau / 8.0))
                r = run(Scenario(detect_range=20.0, bearing=HEAD_ON, params=p))
                self.assertGreater(len(r.trace), 1)


class TestAllCombinationsRun(unittest.TestCase):
    def test_every_policy_and_method_pair_completes(self):
        methods = [cold(), warm(), blend(0.5), filtered_blend(0.5)]
        for policy in WAIT_POLICIES:
            for m in methods:
                with self.subTest(policy=policy, method=m.name):
                    r = run(Scenario(detect_range=22.0, bearing=BROADSIDE,
                                     policy=policy, startup=m, params=FAST))
                    self.assertGreater(len(r.trace), 1)

    def test_the_basic_scenario_set_runs(self):
        for s in basic_set(params=FAST):
            with self.subTest(bearing=s.bearing):
                self.assertGreater(len(run(s).trace), 1)


if __name__ == "__main__":
    unittest.main()
