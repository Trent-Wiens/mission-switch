"""Turn runs into viewer data.

Replay works because the simulator is deterministic: the same parameters give
the same trace every time, so a case only has to be *identified*, never stored.
The grid writes parameters to CSV; this module re-runs any of them on demand.

A case is a dict:

    {"meta": {...}, "metrics": {...}, "t": [...], "ex": [...], ...}

`meta` carries every parameter needed to reproduce it, which is also what the
viewer builds its selectors from.
"""

from __future__ import annotations

import json
import math
from typing import Dict, Iterable, List, Optional

from .params import Params
from .scenarios import BROADSIDE, FROM_BEHIND, HEAD_ON
from .sim import Scenario, run
from .startup import blend, cold, filtered_blend, warm

VIEW_HZ = 20.0

FACTORIES = {
    "cold": cold,
    "warm": warm,
    "blend": lambda: blend(0.5),
    "filtered": lambda: filtered_blend(0.5),
}
METHOD_LABELS = {
    "cold": "Cold start",
    "warm": "Warm start",
    "blend": "Blend 0.5 s",
    "filtered": "Filtered 0.5 s",
}

#: Bearings that get a name instead of a number of degrees.
NAMED_BEARINGS = {
    0.0: "Head-on",
    90.0: "Broadside",
    180.0: "From behind",
    270.0: "Broadside (right)",
}


def bearing_label(deg: float) -> str:
    key = round(deg % 360.0, 1)
    if key in NAMED_BEARINGS:
        return NAMED_BEARINGS[key]
    return f"{key:g}°"


def case_label(meta: dict) -> str:
    return (
        f"{bearing_label(meta['bearing_deg'])}, {meta['detect_range']:g} m, "
        f"{METHOD_LABELS[meta['method']]}, h={meta['h']:g}, "
        f"τ={meta['tau']:g}, {meta['policy']}"
    )


def r2(x: float) -> float:
    return round(x, 2)


def build_case(
    policy: str,
    h: float,
    tau: float,
    method: str,
    bearing_deg: float,
    detect_range: float,
    v_flee: Optional[float] = 8.0,
    window: float = 14.0,
    dt: float = 0.005,
    note: str = "",
) -> dict:
    """Re-run one case and package it for the viewer."""
    params = Params(lag=True, tau=tau, dt=dt, v_max=v_flee, t_eval=window)
    scenario = Scenario(
        detect_range=detect_range,
        bearing=math.radians(bearing_deg),
        h=h,
        policy=policy,
        startup=FACTORIES[method](),
        v_flee=v_flee,
        params=params,
    )
    result = run(scenario)

    stride = max(1, int(round((1.0 / VIEW_HZ) / dt)))
    samples = result.trace[::stride]
    if samples[-1] is not result.trace[-1]:
        samples.append(result.trace[-1])

    meta = {
        "policy": policy,
        "h": h,
        "tau": tau,
        "method": method,
        "bearing_deg": round(bearing_deg, 1),
        "detect_range": round(detect_range, 2),
        "vFlee": v_flee,
        "note": note,
    }
    margin = result.min_margin_after_commit

    return {
        "meta": meta,
        "label": case_label(meta),
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
            # A case caught before h never hands over, so every metric below
            # that describes the handover is a default rather than a measured
            # zero. Callers must check this before reporting any of them.
            "committed": result.committed,
            "tCapture": None if result.t_capture is None else r2(result.t_capture),
            "cmdVelJump": r2(result.cmd_vel_jump),
            "cmdAccelJump": r2(result.cmd_accel_jump),
            "peakCmdSlew": r2(result.peak_cmd_slew),
            "peakAccel": r2(result.peak_accel),
            "minMargin": None if math.isnan(margin) else r2(margin),
            "minDistance": r2(result.min_distance),
            "certAtCommit": result.cert_held_at_commit,
            "falseCert": result.false_certificate,
            "surveyingAtCommit": result.surveying_at_commit,
            "geofenceBreach": result.geofence_breach,
        },
    }


def build_payload(cases: List[dict], title: str = "", subtitle: str = "") -> dict:
    return {
        "title": title,
        "subtitle": subtitle,
        "config": {
            "viewHz": VIEW_HZ,
            "surveilRadius": 30.0,
            "geofenceRadius": 80.0,
            "captureRadius": 3.0,
            "aMax": 4.0,
            "vPursuer": 5.0,
            "surveilSpeed": 3.0,
        },
        "methodLabels": METHOD_LABELS,
        "cases": cases,
    }


def write_payload(payload: dict, js_path: str, json_path: Optional[str] = None) -> int:
    blob = json.dumps(payload, separators=(",", ":"))
    with open(js_path, "w") as fh:
        fh.write("window.RUNS = " + blob + ";\n")
    if json_path:
        with open(json_path, "w") as fh:
            fh.write(blob)
    return len(blob)
