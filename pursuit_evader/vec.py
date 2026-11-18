"""Minimal 2-D vector type. Standard library only."""

from __future__ import annotations

import math
from typing import Iterator, NamedTuple


class Vec2(NamedTuple):
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, k: float) -> "Vec2":
        return Vec2(self.x * k, self.y * k)

    __rmul__ = __mul__

    def __truediv__(self, k: float) -> "Vec2":
        return Vec2(self.x / k, self.y / k)

    def __neg__(self) -> "Vec2":
        return Vec2(-self.x, -self.y)

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vec2") -> float:
        """Scalar z-component of the 3-D cross product."""
        return self.x * other.y - self.y * other.x

    @property
    def norm(self) -> float:
        return math.hypot(self.x, self.y)

    @property
    def angle(self) -> float:
        """Heading in radians, measured counter-clockwise from +x."""
        return math.atan2(self.y, self.x)

    def unit(self) -> "Vec2":
        """Unit vector. Returns the zero vector for a zero-length input."""
        n = self.norm
        if n == 0.0:
            return ZERO
        return Vec2(self.x / n, self.y / n)

    def perp(self) -> "Vec2":
        """Rotate 90 degrees counter-clockwise."""
        return Vec2(-self.y, self.x)

    def rotated(self, theta: float) -> "Vec2":
        c, s = math.cos(theta), math.sin(theta)
        return Vec2(self.x * c - self.y * s, self.x * s + self.y * c)

    def clamp_norm(self, limit: float) -> "Vec2":
        """Scale down to `limit` if longer. Direction is preserved."""
        n = self.norm
        if n <= limit or n == 0.0:
            return self
        return Vec2(self.x * limit / n, self.y * limit / n)


ZERO = Vec2(0.0, 0.0)


def from_polar(radius: float, theta: float) -> Vec2:
    return Vec2(radius * math.cos(theta), radius * math.sin(theta))


def distance(a: Vec2, b: Vec2) -> float:
    return (a - b).norm
