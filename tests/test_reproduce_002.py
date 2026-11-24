"""Step 1 of the handoff: reproduce experiment 002.

Experiment 002 ran the *old* plant -- the escape controller had full, immediate
authority -- so these tests pin `lag=False`.

Its headline required-detection-range figures are the worst case over all
bearings, and the worst bearing is head-on, so the numeric checks below use
head-on.  This build reproduces 11.1/11.1/12.1 m for prepare against the
reported 11/11/11, and 15.1/19.0/26.8 m for defer against the reported
15/20/28.  Tolerances are wide enough to absorb controller gains the handoff
does not pin down, but tight enough that a real regression fails them.

One finding does *not* reproduce: experiment 002 reported stop-first needing
1.9 to 4.3 times what prepare needs when approached from behind.  This build
gets 1.4 to 2.0.  See `test_stop_first_ratio_is_below_what_002_reported`.
"""

import math
import unittest

from pursuit_evader.params import Params
from pursuit_evader.scenarios import FROM_BEHIND, H_VALUES, HEAD_ON
from pursuit_evader.startup import warm
from pursuit_evader.sweep import required_detection_range, verify_monotone

from pursuit_evader.presets import LEGACY

#: Warm start is the closest match to "ideal instant switch" with this
#: controller, since the old toy had no controller memory to initialise.
IDEAL = dict(startup=warm(), params=LEGACY)


def need(policy, h, bearing):
    return required_detection_range(
        bearing=bearing, policy=policy, h=h, **IDEAL
    )


class TestEscapeIsMonotoneInRange(unittest.TestCase):
    """The bisection in `required_detection_range` assumes this."""

    def test_monotone_head_on(self):
        self.assertTrue(verify_monotone(HEAD_ON, policy="prepare", h=1.0, **IDEAL))

    def test_monotone_from_behind(self):
        self.assertTrue(verify_monotone(FROM_BEHIND, policy="prepare", h=1.0, **IDEAL))


class TestHeadOnIsTheWorstBearing(unittest.TestCase):
    """Experiment 002's headline numbers are worst-over-bearings."""

    def test_no_bearing_needs_more_warning_than_head_on(self):
        worst = need("prepare", 1.0, HEAD_ON)
        for k in range(0, 24, 3):
            b = 2.0 * math.pi * k / 24.0
            with self.subTest(bearing_deg=round(math.degrees(b))):
                self.assertLessEqual(need("prepare", 1.0, b), worst + 0.3)


class TestReportedNumbers(unittest.TestCase):
    """The 11/11/11 and 15/20/28 metre figures from experiment 002."""

    def test_prepare_needs_about_eleven_metres_at_every_h(self):
        for h, reported in zip(H_VALUES, (11.0, 11.0, 11.0)):
            with self.subTest(h=h):
                self.assertAlmostEqual(need("prepare", h, HEAD_ON), reported, delta=1.5)

    def test_defer_reproduces_fifteen_twenty_twentyeight(self):
        for h, reported in zip(H_VALUES, (15.0, 20.0, 28.0)):
            with self.subTest(h=h):
                self.assertAlmostEqual(need("defer", h, HEAD_ON), reported, delta=2.5)


class TestPrepareBeatsTheAlternatives(unittest.TestCase):
    """Finding: preparing at once dominates deferring and braking to a hover."""

    def test_prepare_needs_no_more_warning_than_defer(self):
        for h in H_VALUES:
            for bearing in (HEAD_ON, FROM_BEHIND):
                with self.subTest(h=h, bearing=bearing):
                    self.assertLessEqual(
                        need("prepare", h, bearing), need("defer", h, bearing) + 1e-9
                    )

    def test_prepare_needs_no_more_warning_than_stop_first(self):
        for h in H_VALUES:
            for bearing in (HEAD_ON, FROM_BEHIND):
                with self.subTest(h=h, bearing=bearing):
                    self.assertLessEqual(
                        need("prepare", h, bearing),
                        need("stop_first", h, bearing) + 1e-9,
                    )

    def test_stop_first_is_a_poor_default_from_behind(self):
        p = need("prepare", 1.0, FROM_BEHIND)
        s = need("stop_first", 1.0, FROM_BEHIND)
        self.assertGreater(s, 1.3 * p)

    def test_stop_first_ratio_is_below_what_002_reported(self):
        """Honest non-reproduction, pinned so it is not silently lost.

        Experiment 002 reported 1.9x to 4.3x. This build gets roughly 1.4x to
        2.0x: the same direction, a smaller effect. The likely cause is the
        escape controller's gains, which the handoff does not specify.
        """
        ratios = [
            need("stop_first", h, FROM_BEHIND) / need("prepare", h, FROM_BEHIND)
            for h in H_VALUES
        ]
        self.assertTrue(all(r > 1.0 for r in ratios), ratios)
        self.assertLess(max(ratios), 4.3)


class TestWaitingCostsWarning(unittest.TestCase):
    """Finding: prepare is flat in h, defer is not."""

    def test_prepare_is_insensitive_to_the_wait(self):
        needs = [need("prepare", h, HEAD_ON) for h in H_VALUES]
        self.assertLess(max(needs) - min(needs), 3.0)

    def test_defer_gets_worse_as_the_wait_grows(self):
        needs = [need("defer", h, HEAD_ON) for h in H_VALUES]
        self.assertEqual(needs, sorted(needs))
        self.assertGreater(needs[-1], needs[0] + 5.0)

    def test_defer_and_prepare_coincide_from_behind(self):
        """Geometric, not a bug: fleeing astern is what surveying already does."""
        for h in H_VALUES:
            with self.subTest(h=h):
                self.assertAlmostEqual(
                    need("defer", h, FROM_BEHIND),
                    need("prepare", h, FROM_BEHIND),
                    delta=0.3,
                )


class TestForecastForward(unittest.TestCase):
    """Finding: the forecast check kept surveying often and never lost a case."""

    def test_never_needs_more_warning_than_deferring(self):
        for h in H_VALUES:
            for bearing in (HEAD_ON, FROM_BEHIND):
                with self.subTest(h=h, bearing=bearing):
                    self.assertLessEqual(
                        need("forecast_forward", h, bearing),
                        need("defer", h, bearing) + 1e-9,
                    )

    def test_matches_prepare_where_surveying_cannot_be_proved_safe(self):
        for h in H_VALUES:
            with self.subTest(h=h):
                self.assertAlmostEqual(
                    need("forecast_forward", h, HEAD_ON),
                    need("prepare", h, HEAD_ON),
                    delta=1.0,
                )


if __name__ == "__main__":
    unittest.main()
