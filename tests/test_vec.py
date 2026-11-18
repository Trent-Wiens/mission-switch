import math
import unittest

from pursuit_evader.vec import ZERO, Vec2, distance, from_polar


class TestVec2(unittest.TestCase):
    def test_arithmetic(self):
        a, b = Vec2(1.0, 2.0), Vec2(3.0, -1.0)
        self.assertEqual(a + b, Vec2(4.0, 1.0))
        self.assertEqual(a - b, Vec2(-2.0, 3.0))
        self.assertEqual(a * 2.0, Vec2(2.0, 4.0))
        self.assertEqual(2.0 * a, Vec2(2.0, 4.0))
        self.assertEqual(a / 2.0, Vec2(0.5, 1.0))
        self.assertEqual(-a, Vec2(-1.0, -2.0))

    def test_dot_and_cross(self):
        self.assertAlmostEqual(Vec2(1.0, 0.0).dot(Vec2(0.0, 1.0)), 0.0)
        self.assertAlmostEqual(Vec2(2.0, 3.0).dot(Vec2(4.0, 5.0)), 23.0)
        self.assertAlmostEqual(Vec2(1.0, 0.0).cross(Vec2(0.0, 1.0)), 1.0)

    def test_norm_and_unit(self):
        self.assertAlmostEqual(Vec2(3.0, 4.0).norm, 5.0)
        self.assertAlmostEqual(Vec2(3.0, 4.0).unit().norm, 1.0)

    def test_unit_of_zero_is_zero(self):
        """A zero vector has no direction; returning zero keeps callers safe."""
        self.assertEqual(ZERO.unit(), ZERO)

    def test_perp_is_ccw_and_orthogonal(self):
        v = Vec2(1.0, 0.0)
        self.assertEqual(v.perp(), Vec2(0.0, 1.0))
        self.assertAlmostEqual(v.dot(v.perp()), 0.0)

    def test_rotated(self):
        v = Vec2(1.0, 0.0).rotated(math.pi / 2.0)
        self.assertAlmostEqual(v.x, 0.0)
        self.assertAlmostEqual(v.y, 1.0)

    def test_clamp_norm(self):
        self.assertAlmostEqual(Vec2(3.0, 4.0).clamp_norm(2.5).norm, 2.5)
        # Shorter than the limit is left untouched.
        self.assertEqual(Vec2(1.0, 0.0).clamp_norm(5.0), Vec2(1.0, 0.0))
        # Direction survives the clamp.
        clamped = Vec2(3.0, 4.0).clamp_norm(1.0)
        self.assertAlmostEqual(clamped.x, 0.6)
        self.assertAlmostEqual(clamped.y, 0.8)

    def test_from_polar_and_distance(self):
        v = from_polar(5.0, 0.0)
        self.assertAlmostEqual(v.x, 5.0)
        self.assertAlmostEqual(v.y, 0.0)
        self.assertAlmostEqual(distance(Vec2(0.0, 0.0), Vec2(3.0, 4.0)), 5.0)

    def test_angle(self):
        self.assertAlmostEqual(Vec2(0.0, 1.0).angle, math.pi / 2.0)


if __name__ == "__main__":
    unittest.main()
