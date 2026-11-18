#!/usr/bin/env python3
"""Export runs for the interactive viewer.

    python3 export_viz.py

Writes viewer/runs.js (loaded by viewer/index.html via a script tag) and
../Images/runs.json (the same payload, for any other tooling).

Columnar arrays, rounded, downsampled to VIEW_HZ -- the viewer only needs
enough resolution to animate smoothly, not the solver's step size.
"""

from __future__ import annotations

import json
import math
import os
import sys

from pursuit_evader.params import Params
from pursuit_evader.scenarios import BROADSIDE, FROM_BEHIND, HEAD_ON
from pursuit_evader.sim import Scenario, run
from pursuit_evader.startup import blend, cold, filtered_blend, warm

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(os.path.dirname(HERE), "Images")

VIEW_HZ = 20.0
WINDOW = 12.0           # s of simulated time to export
H = 1.0                 # wait before commit
TAU = 0.3
V_FLEE = 8.0

BEARINGS = [
    ("head_on", "Head-on", HEAD_ON),
    ("broadside", "Broadside", BROADSIDE),
    ("behind", "From behind", FROM_BEHIND),
]
RANGES = [18.0, 25.0, 40.0]
METHODS = [
    ("cold", "Cold start", cold),
    ("warm", "Warm start", warm),
    ("blend", "Blend 0.5 s", lambda: blend(0.5)),
    ("filtered", "Filtered 0.5 s", lambda: filtered_blend(0.5)),
]


def r2(x: float) -> float:
    return round(x, 2)


def export_run(bearing: float, detect_range: float, make_method) -> dict:
    p = Params(lag=True, tau=TAU, dt=0.005, v_max=V_FLEE, t_eval=WINDOW)
    result = run(Scenario(
        detect_range=detect_range, bearing=bearing, h=H, policy="defer",
        startup=make_method(), params=p, v_flee=V_FLEE,
    ))

    stride = max(1, int(round((1.0 / VIEW_HZ) / p.dt)))
    samples = result.trace[::stride]
    if samples[-1] is not result.trace[-1]:
        samples.append(result.trace[-1])

    return {
        "t": [r2(s.t) for s in samples],
        "ex": [r2(s.evader_pos.x) for s in samples],
        "ey": [r2(s.evader_pos.y) for s in samples],
        "px": [r2(s.pursuer_pos.x) for s in samples],
        "py": [r2(s.pursuer_pos.y) for s in samples],
        "cmd": [r2(s.cmd.norm) for s in samples],
        "spd": [r2(s.evader_vel.norm) for s in samples],
        "acc": [r2(s.evader_accel.norm) for s in samples],
        "mar": [r2(s.margin) for s in samples],
        "dist": [r2(s.distance) for s in samples],
        "metrics": {
            "captured": result.captured,
            "tCapture": None if result.t_capture is None else r2(result.t_capture),
            "cmdVelJump": r2(result.cmd_vel_jump),
            "cmdAccelJump": r2(result.cmd_accel_jump),
            "peakCmdSlew": r2(result.peak_cmd_slew),
            "minMargin": (None if math.isnan(result.min_margin_after_commit)
                          else r2(result.min_margin_after_commit)),
            "minDistance": r2(result.min_distance),
            "certAtCommit": result.cert_held_at_commit,
            "falseCert": result.false_certificate,
        },
    }


def main() -> int:
    runs = {}
    for bkey, _, bearing in BEARINGS:
        for rng in RANGES:
            for mkey, _, make in METHODS:
                runs[f"{bkey}|{rng:g}|{mkey}"] = export_run(bearing, rng, make)

    payload = {
        "config": {
            "h": H, "tau": TAU, "vFlee": V_FLEE, "viewHz": VIEW_HZ,
            "surveilRadius": 30.0, "geofenceRadius": 80.0,
            "captureRadius": 3.0, "aMax": 4.0,
            "vPursuer": 5.0, "surveilSpeed": 3.0,
        },
        "bearings": [{"key": k, "label": l} for k, l, _ in BEARINGS],
        "ranges": RANGES,
        "methods": [{"key": k, "label": l} for k, l, _ in METHODS],
        "runs": runs,
    }

    blob = json.dumps(payload, separators=(",", ":"))

    os.makedirs(IMAGES, exist_ok=True)
    json_path = os.path.join(IMAGES, "runs.json")
    with open(json_path, "w") as fh:
        fh.write(blob)

    viewer = os.path.join(HERE, "viewer")
    os.makedirs(viewer, exist_ok=True)
    js_path = os.path.join(viewer, "runs.js")
    with open(js_path, "w") as fh:
        fh.write("window.RUNS = " + blob + ";\n")

    size_kb = os.path.getsize(js_path) / 1024.0
    print(f"wrote {len(runs)} runs ({size_kb:.0f} KB)")
    print(f"  {js_path}")
    print(f"  {json_path}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.exit(main())
