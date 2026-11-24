#!/usr/bin/env python3
"""Build a set of runs to browse side by side in the viewer.

`replay.py` pulls cases *out of the grid*.  This one generates a deliberate
cross-product instead, so it needs no grid and can cover combinations the
sweep never ran -- `stop_first`, for instance.

    python3 viewset.py combos     4 methods x 4 policies   (16 runs)
    python3 viewset.py params     3 values of h x 3 of tau (9 runs)
    python3 viewset.py geometry   bearings x ranges
    python3 viewset.py custom --vary method,tau

Everything not being varied is pinned with --h / --tau / --method / --policy /
--bearing / --range, so any difference you see is caused by the thing that
changed.

The viewer builds its controls from whatever is in the file, so each varying
dimension turns into its own selector automatically.
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser

from pursuit_evader.presets import V_FLEE
from pursuit_evader.viz import (
    load_grid_cells,
    METHOD_LABELS, build_case, build_payload, write_payload,
)

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(os.path.dirname(HERE), "Images")
VIEWER = os.path.join(HERE, "viewer")

METHODS = ("cold", "warm", "blend", "filtered")
POLICIES = ("defer", "prepare", "stop_first", "forecast_forward")
H_VALUES = (0.5, 1.0, 2.0)
TAU_VALUES = (0.1, 0.3, 0.5)

#: What `custom --vary` accepts, and the values each dimension sweeps.
DIMENSIONS = {
    "method": METHODS,
    "policy": POLICIES,
    "h": H_VALUES,
    "tau": TAU_VALUES,
}

MAX_CASES = 200     # a guard, not a limit anyone should hit by accident


def pinned(args):
    return {
        "policy": args.policy, "h": args.h, "tau": args.tau,
        "method": args.method, "bearing_deg": args.bearing,
        "detect_range": args.range,
    }


def spread(args, dims):
    """Every combination of `dims`, with everything else pinned."""
    specs = [dict(pinned(args))]
    for dim in dims:
        values = (DIMENSIONS[dim] if dim in DIMENSIONS
                  else getattr(args, dim + "_values"))
        grown = []
        for base in specs:
            for v in values:
                grown.append(dict(base, **{dim: v}))
        specs = grown
    return specs


def build_specs(args):
    if args.cmd == "combos":
        return spread(args, ["policy", "method"]), "policy x method"
    if args.cmd == "params":
        return spread(args, ["h", "tau"]), "h x tau"
    if args.cmd == "geometry":
        bearings = [round(360.0 * i / args.bearings, 1) for i in range(args.bearings)]
        ranges = [round(args.rmin + (args.rmax - args.rmin) * i / max(1, args.ranges - 1), 1)
                  for i in range(args.ranges)]
        specs = []
        for b in bearings:
            for r in ranges:
                specs.append(dict(pinned(args), bearing_deg=b, detect_range=r))
        return specs, "bearing x range"
    # custom
    dims = [d.strip() for d in args.vary.split(",") if d.strip()]
    bad = [d for d in dims if d not in DIMENSIONS]
    if bad:
        sys.exit(f"cannot vary {bad} -- choose from {sorted(DIMENSIONS)}")
    return spread(args, dims), " x ".join(dims)


DIM_COLS = (("policy", "policy", 18), ("method", "method", 11),
            ("h", "h", 6), ("tau", "tau", 6),
            ("bearing_deg", "bearing", 9), ("detect_range", "range", 8))


def varying_dims(cases):
    """Which dimensions actually differ across the set."""
    out = []
    for key, label, width in DIM_COLS:
        if len({c["meta"][key] for c in cases}) > 1:
            out.append((key, label, width))
    return out


def cell(value):
    return value if isinstance(value, str) else f"{value:g}"


def summary(cases, dims):
    """Show the varying dimensions as columns.

    Relying on the case label truncates exactly the part that differs, since
    the label puts policy last and everything else is identical.
    """
    head = "".join(f"{label:<{w}}" for _, label, w in dims)
    print()
    print(head + f"{'outcome':>12}{'cmd dv':>9}{'min margin':>12}{'cert':>11}")
    print("-" * (len(head) + 44))
    for c in cases:
        m = c["metrics"]
        outcome = f"caught {m['tCapture']:g}s" if m["captured"] else "escaped"
        row = "".join(f"{cell(c['meta'][k]):<{w}}" for k, _, w in dims)
        if not m["committed"]:
            # Caught before h: there was no handover, so the handover columns
            # would be reporting defaults, not measurements.
            print(row + f"{outcome:>12}{'--':>9}{'--':>12}{'no switch':>11}")
            continue
        margin = "n/a" if m["minMargin"] is None else f"{m['minMargin']:.2f}"
        cert = "FALSE" if m["falseCert"] else ("held" if m["certAtCommit"] else "refused")
        print(row + f"{outcome:>12}{m['cmdVelJump']:>9.2f}{margin:>12}{cert:>11}")
    print("-" * (len(head) + 44))
    caught = sum(c["metrics"]["captured"] for c in cases)
    false = sum(c["metrics"]["falseCert"] for c in cases)
    never = sum(0 if c["metrics"]["committed"] else 1 for c in cases)
    line = f"{len(cases)} runs   {caught} captured   {false} false certificates"
    if never:
        line += f"   {never} caught before the switch"
    print(line)


def main(argv) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("combos", "params", "geometry", "custom"))

    pin = ap.add_argument_group("pin everything not being varied")
    pin.add_argument("--policy", default="defer", choices=POLICIES)
    pin.add_argument("--method", default="cold", choices=METHODS)
    pin.add_argument("--h", type=float, default=1.0)
    pin.add_argument("--tau", type=float, default=0.3)
    pin.add_argument("--bearing", type=float, default=0.0, help="degrees")
    pin.add_argument("--range", type=float, default=20.0, help="metres")

    geo = ap.add_argument_group("geometry only")
    geo.add_argument("--bearings", type=int, default=8)
    geo.add_argument("--ranges", type=int, default=5)
    geo.add_argument("--rmin", type=float, default=10.0)
    geo.add_argument("--rmax", type=float, default=35.0)

    cus = ap.add_argument_group("custom only")
    cus.add_argument("--vary", default="method",
                     help="comma-separated: method, policy, h, tau")

    out = ap.add_argument_group("output")
    out.add_argument("--vflee", type=float, default=V_FLEE)
    out.add_argument("--window", type=float, default=14.0)
    out.add_argument("--no-open", action="store_true")
    out.add_argument("--quiet", action="store_true",
                     help="skip the summary table")

    args = ap.parse_args(argv)
    specs, what = build_specs(args)

    if len(specs) > MAX_CASES:
        sys.exit(f"{len(specs)} runs is more than the {MAX_CASES} guard -- "
                 "narrow it down, or raise MAX_CASES if you mean it")

    print(f"building {len(specs)} runs: {what}", file=sys.stderr)
    cases = [
        build_case(
            policy=s["policy"], h=s["h"], tau=s["tau"], method=s["method"],
            bearing_deg=s["bearing_deg"], detect_range=s["detect_range"],
            v_flee=args.vflee, window=args.window,
        )
        for s in specs
    ]

    dims = varying_dims(cases)
    if not args.quiet:
        summary(cases, dims)

    varying_keys = {k for k, _, _ in dims}
    fixed = [f"{label} {cell(cases[0]['meta'][key])}"
             for key, label, _ in DIM_COLS if key not in varying_keys]

    payload = build_payload(
        cases,
        grid_cells=load_grid_cells(IMAGES),
        title=f"Varying {what}",
        subtitle=(f"{len(cases)} runs. Fixed: " + ", ".join(fixed) +
                  ". Every varying dimension gets its own selector."),
    )
    os.makedirs(VIEWER, exist_ok=True)
    os.makedirs(IMAGES, exist_ok=True)
    js = os.path.join(VIEWER, "runs.js")
    size = write_payload(payload, js, os.path.join(IMAGES, "runs.json"))
    print(f"\nwrote {len(cases)} cases to {js} ({size/1024:.0f} KB)")

    if not args.no_open:
        webbrowser.open("file://" + os.path.join(VIEWER, "index.html"))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.exit(main(sys.argv[1:]))
