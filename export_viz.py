#!/usr/bin/env python3
"""Export a cross-product of cases for the viewer.

    python3 export_viz.py

Writes viewer/runs.js and ../Images/runs.json.  The viewer builds its controls
from whatever is in here, so widening a list below gives you another selector
with no change to the page.

For arbitrary, hand-picked cases -- the worst jolt, every false certificate, a
set for a talk -- use `replay.py --viewer` instead.
"""

from __future__ import annotations

import os
import sys

from pursuit_evader.viz import build_case, build_payload, write_payload

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(os.path.dirname(HERE), "Images")
VIEWER = os.path.join(HERE, "viewer")

POLICIES = ["defer"]
BEARINGS = [0.0, 90.0, 180.0]
RANGES = [18.0, 25.0, 40.0]
H_VALUES = [0.5, 1.0, 2.0]
TAU_VALUES = [0.1, 0.3, 0.5]
METHODS = ["cold", "warm", "blend", "filtered"]
V_FLEE = 8.0
WINDOW = 14.0


def main() -> int:
    cases = []
    for policy in POLICIES:
        for bearing in BEARINGS:
            for rng in RANGES:
                for h in H_VALUES:
                    for tau in TAU_VALUES:
                        for method in METHODS:
                            cases.append(build_case(
                                policy=policy, h=h, tau=tau, method=method,
                                bearing_deg=bearing, detect_range=rng,
                                v_flee=V_FLEE, window=WINDOW,
                            ))

    payload = build_payload(
        cases,
        title="Starting the escape controller without a jolt",
        subtitle=("Replay of the handover instant. The drone surveys until a "
                  "pursuer is detected at t = 0, waits out the hold, then "
                  "commits to an escape controller."),
    )

    os.makedirs(VIEWER, exist_ok=True)
    os.makedirs(IMAGES, exist_ok=True)
    js = os.path.join(VIEWER, "runs.js")
    size = write_payload(payload, js, os.path.join(IMAGES, "runs.json"))

    print(f"wrote {len(cases)} cases ({size/1024:.0f} KB)")
    print(f"  {js}")
    print(f"  {os.path.join(IMAGES, 'runs.json')}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.exit(main())
