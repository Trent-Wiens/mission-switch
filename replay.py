#!/usr/bin/env python3
"""Replay any case from the grid.

The simulator is deterministic, so a case only has to be *identified*, never
stored.  The grid CSV holds parameters; this re-runs them and gives you the
trace back in whatever form you want.

    # pick cases out of the grid by what makes them interesting
    python3 replay.py --worst-jolt
    python3 replay.py --false-cert
    python3 replay.py --captures --limit 6
    python3 replay.py --surveying-vs-turned

    # or name one directly, no grid needed
    python3 replay.py --policy defer --h 1 --tau 0.3 --method cold \
                      --bearing 0 --range 18

    # what to do with them (combine freely; --table is the default)
    --table      print the trace, one line per 0.5 s
    --csv        write a trace CSV per case into Images/
    --viewer     bundle them into viewer/runs.js and open the viewer
    --compare    include all four methods for each selected case

Selection flags read Images/grid_results.csv, so run `python3 grid.py` first.
Direct flags work without it.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import webbrowser

from pursuit_evader.viz import (
    METHOD_LABELS, bearing_label, build_case, build_payload, write_payload,
)

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(os.path.dirname(HERE), "Images")
GRID_CSV = os.path.join(IMAGES, "grid_results.csv")
VIEWER_DIR = os.path.join(HERE, "viewer")

METHOD_ORDER = ("cold", "warm", "blend", "filtered")


def load_grid():
    if not os.path.exists(GRID_CSV):
        sys.exit(
            f"no grid results at {GRID_CSV}\n"
            "run `python3 grid.py` first, or name a case directly with "
            "--policy/--h/--tau/--method/--bearing/--range"
        )
    rows = []
    with open(GRID_CSV) as fh:
        for row in csv.DictReader(fh):
            for k in ("h", "tau", "detect_range", "bearing_deg", "cmd_vel_jump",
                      "cmd_accel_jump", "peak_cmd_accel", "peak_cmd_slew",
                      "peak_accel", "min_distance"):
                row[k] = float(row[k])
            for k in ("captured", "captured_after_commit", "cert_held_at_commit",
                      "false_certificate", "surveying_at_commit", "geofence_breach"):
                row[k] = int(row[k])
            m = row["min_margin_after_commit"]
            row["min_margin_after_commit"] = None if m == "" else float(m)
            rows.append(row)
    return rows


def spec_of(row, note=""):
    return {
        "policy": row["policy"], "h": row["h"], "tau": row["tau"],
        "method": row["method"], "bearing_deg": row["bearing_deg"],
        "detect_range": row["detect_range"], "note": note,
    }


def select(args):
    """Return a list of case specs."""
    if args.policy is not None:
        return [{
            "policy": args.policy, "h": args.h, "tau": args.tau,
            "method": args.method, "bearing_deg": args.bearing,
            "detect_range": args.range, "note": "",
        }]

    rows = load_grid()
    picked = []

    if args.worst_jolt:
        rows.sort(key=lambda r: -r["cmd_vel_jump"])
        picked += [spec_of(r, "largest command jump in the grid")
                   for r in rows[:args.limit]]

    if args.false_cert:
        bad = [r for r in rows if r["false_certificate"]]
        if not bad:
            print("no false certificates anywhere in the grid -- that is the finding")
        picked += [spec_of(r, "certificate held at commit, then captured")
                   for r in bad[:args.limit]]

    if args.captures:
        caught = [r for r in rows if r["captured_after_commit"]]
        caught.sort(key=lambda r: r["min_distance"])
        picked += [spec_of(r, "captured after commit") for r in caught[:args.limit]]

    if args.worst_margin:
        have = [r for r in rows if r["min_margin_after_commit"] is not None]
        have.sort(key=lambda r: r["min_margin_after_commit"])
        picked += [spec_of(r, "smallest margin after commit")
                   for r in have[:args.limit]]

    if args.surveying_vs_turned:
        ff = [r for r in rows if r["policy"] == "forecast_forward"]
        surv = sorted([r for r in ff if r["surveying_at_commit"]],
                      key=lambda r: -r["cmd_vel_jump"])
        turn = sorted([r for r in ff if not r["surveying_at_commit"]],
                      key=lambda r: -r["cmd_vel_jump"])
        picked += [spec_of(r, "forecast_forward, still surveying at commit")
                   for r in surv[:max(1, args.limit // 2)]]
        picked += [spec_of(r, "forecast_forward, already turned at commit")
                   for r in turn[:max(1, args.limit // 2)]]

    if not picked:
        sys.exit("nothing selected -- pass a selection flag, or see --help")

    # Drop duplicates, keeping the first note.
    seen, out = set(), []
    for s in picked:
        key = (s["policy"], s["h"], s["tau"], s["method"],
               round(s["bearing_deg"], 1), round(s["detect_range"], 2))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def expand_methods(specs):
    """One spec per method, for each selected case."""
    out, seen = [], set()
    for s in specs:
        for m in METHOD_ORDER:
            t = dict(s, method=m)
            key = (t["policy"], t["h"], t["tau"], m,
                   round(t["bearing_deg"], 1), round(t["detect_range"], 2))
            if key not in seen:
                seen.add(key)
                out.append(t)
    return out


def print_table(case, every_s=0.5):
    m = case["metrics"]
    print()
    print("=" * 78)
    print(case["label"])
    if case["meta"]["note"]:
        print("  " + case["meta"]["note"])
    print("=" * 78)
    outcome = f"CAPTURED at {m['tCapture']} s" if m["captured"] else "escaped"
    margin = "n/a" if m["minMargin"] is None else f"{m['minMargin']:.2f} m"
    cert = "FALSE" if m["falseCert"] else ("held" if m["certAtCommit"] else "refused")
    at_commit = "surveying" if m["surveyingAtCommit"] else "already turned"

    print(f"  outcome      {outcome}")
    print(f"  cmd jump     {m['cmdVelJump']:.2f} m/s"
          f"      demanded accel jump {m['cmdAccelJump']:.1f} m/s^2")
    print(f"  min margin   {margin}"
          f"      closest {m['minDistance']:.1f} m")
    print(f"  certificate  {cert} at commit"
          f"      at commit: {at_commit}")
    print()
    print(f"  {'t':>6} {'speed':>8} {'cmd':>8} {'accel':>8} {'margin':>9} {'gap':>8}  phase")

    stride = max(1, int(round(every_s * len(case["t"]) / max(1e-9, case["t"][-1]))))
    for i in range(0, len(case["t"]), stride):
        t = case["t"][i]
        phase = "survey/wait" if t < case["meta"]["h"] else "escape"
        mark = "  <-- COMMIT" if abs(t - case["meta"]["h"]) < 1e-9 else ""
        print(f"  {t:6.2f} {case['spd'][i]:8.2f} {case['cmd'][i]:8.2f} "
              f"{case['acc'][i]:8.2f} {case['mar'][i]:9.2f} {case['dist'][i]:8.1f}  "
              f"{phase}{mark}")


def write_trace_csv(case):
    meta = case["meta"]
    name = (f"replay_{meta['policy']}_h{meta['h']:g}_tau{meta['tau']:g}_"
            f"{meta['method']}_b{meta['bearing_deg']:g}_r{meta['detect_range']:g}.csv")
    path = os.path.join(IMAGES, name)
    os.makedirs(IMAGES, exist_ok=True)
    cols = ["t", "ex", "ey", "px", "py", "cmd", "spd", "acc", "mar", "dist"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "evader_x", "evader_y", "pursuer_x", "pursuer_y",
                    "cmd_speed", "actual_speed", "accel", "margin", "gap"])
        for i in range(len(case["t"])):
            w.writerow([case[c][i] for c in cols])
    return path


def main(argv) -> int:
    ap = argparse.ArgumentParser(
        description="Replay cases from the grid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g = ap.add_argument_group("selection from the grid")
    g.add_argument("--worst-jolt", action="store_true")
    g.add_argument("--false-cert", action="store_true")
    g.add_argument("--captures", action="store_true")
    g.add_argument("--worst-margin", action="store_true")
    g.add_argument("--surveying-vs-turned", action="store_true")
    g.add_argument("--limit", type=int, default=4)

    d = ap.add_argument_group("name a case directly")
    d.add_argument("--policy")
    d.add_argument("--h", type=float, default=1.0)
    d.add_argument("--tau", type=float, default=0.3)
    d.add_argument("--method", default="cold", choices=METHOD_ORDER)
    d.add_argument("--bearing", type=float, default=0.0, help="degrees")
    d.add_argument("--range", type=float, default=20.0, help="metres")

    o = ap.add_argument_group("output")
    o.add_argument("--table", action="store_true")
    o.add_argument("--csv", action="store_true")
    o.add_argument("--viewer", action="store_true")
    o.add_argument("--compare", action="store_true",
                   help="include all four start-up methods per case")
    o.add_argument("--no-open", action="store_true",
                   help="write the viewer bundle but do not open a browser")
    o.add_argument("--vflee", type=float, default=8.0)
    o.add_argument("--window", type=float, default=14.0)

    args = ap.parse_args(argv)
    if not (args.table or args.csv or args.viewer):
        args.table = True

    specs = select(args)
    if args.compare:
        specs = expand_methods(specs)

    print(f"replaying {len(specs)} case(s)", file=sys.stderr)
    cases = [
        build_case(
            policy=s["policy"], h=s["h"], tau=s["tau"], method=s["method"],
            bearing_deg=s["bearing_deg"], detect_range=s["detect_range"],
            v_flee=args.vflee, window=args.window, note=s.get("note", ""),
        )
        for s in specs
    ]

    if args.table:
        for c in cases:
            print_table(c)

    if args.csv:
        print()
        for c in cases:
            print("wrote", write_trace_csv(c))

    if args.viewer:
        payload = build_payload(
            cases,
            title="Switch replay",
            subtitle=(f"{len(cases)} case(s) replayed from the grid. "
                      "Scrub the handover; the plan view and all three charts "
                      "share one cursor."),
        )
        os.makedirs(VIEWER_DIR, exist_ok=True)
        js = os.path.join(VIEWER_DIR, "runs.js")
        size = write_payload(payload, js, os.path.join(IMAGES, "runs.json"))
        print(f"\nwrote {len(cases)} cases to {js} ({size/1024:.0f} KB)")
        if not args.no_open:
            webbrowser.open("file://" + os.path.join(VIEWER_DIR, "index.html"))

    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.exit(main(sys.argv[1:]))
