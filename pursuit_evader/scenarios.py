"""Named scenarios and the grid from step 7 of the handoff.

Bearing convention: the angle of the pursuer as seen from the evader, measured
from the evader's own heading.

    0        pursuer dead ahead          "head-on"
    pi/2     pursuer off the left wing   "broadside"
    pi       pursuer dead astern         "from behind"
"""

from __future__ import annotations

import math
from typing import Iterator, List, Tuple

from .params import DEFAULT, Params
from .sim import Scenario
from .startup import StartupMethod, cold

HEAD_ON = 0.0
BROADSIDE = math.pi / 2.0
FROM_BEHIND = math.pi

#: The grid the results table needs: 24 bearings x 18 ranges.
N_BEARINGS = 24
N_RANGES = 18
RANGE_MIN = 5.0
RANGE_MAX = 40.0

#: Values the handoff asks to sweep.
H_VALUES = (0.5, 1.0, 2.0)
TAU_VALUES = (0.1, 0.3, 0.5)


def bearings(n: int = N_BEARINGS) -> List[float]:
    return [2.0 * math.pi * k / n for k in range(n)]


def ranges(n: int = N_RANGES, lo: float = RANGE_MIN, hi: float = RANGE_MAX) -> List[float]:
    if n == 1:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + step * k for k in range(n)]


def grid(**kwargs) -> Iterator[Scenario]:
    """All 432 detection cases. Extra keyword arguments go to `Scenario`."""
    for b in bearings():
        for r in ranges():
            yield Scenario(detect_range=r, bearing=b, **kwargs)


def head_on(detect_range: float = 15.0, **kwargs) -> Scenario:
    return Scenario(detect_range=detect_range, bearing=HEAD_ON, **kwargs)


def from_behind(detect_range: float = 15.0, **kwargs) -> Scenario:
    return Scenario(detect_range=detect_range, bearing=FROM_BEHIND, **kwargs)


def broadside(detect_range: float = 15.0, **kwargs) -> Scenario:
    return Scenario(detect_range=detect_range, bearing=BROADSIDE, **kwargs)


#: A small set used by the tests and by the smoke run.
def basic_set(**kwargs) -> List[Scenario]:
    return [
        head_on(**kwargs),
        broadside(**kwargs),
        from_behind(**kwargs),
    ]
