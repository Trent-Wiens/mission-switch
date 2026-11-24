#!/usr/bin/env python3
"""Step 7 of the handoff: the full grid, and the section 4 results tables.

    python3 grid.py              run everything, write CSV, print the tables
    python3 grid.py --quick      a coarse grid, for checking the plumbing
    python3 grid.py --tables     re-print the tables from the existing CSV
    python3 grid.py --reqonly    redo just the required-range bisections
    python3 grid.py --tables --markdown > ../Notes/results_table.md

432 detection cases (24 bearings x 18 ranges) x 3 values of h x 3 values of tau
x 4 start-up methods, under three wait policies.  The handoff asks for `prepare`
and `forecast_forward`; `defer` is included as well because the claim being
tested -- that the jolt is largest when the drone switches straight from
surveying -- needs the always-surveying case to be the reference.

Raw per-case rows go to Images/grid_results.csv so the numbers can be re-checked
without re-running the sweep.  Everything else in this project recomputes from
scratch; this is the one place that would be painful to.

ASSUMPTION, NOT A RESULT: the evader's top speed is taken as 8 m/s, because the
handoff never states one.  See Notes/findings_001.md section 4.  Every number
below moves if that changes.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from multiprocessing import Pool

from pursuit_evader.params import Params
from pursuit_evader.presets import DT, T_EVAL, V_FLEE
from pursuit_evader.scenarios import H_VALUES, TAU_VALUES, bearings, ranges
from pursuit_evader.sim import Scenario, run
from pursuit_evader.startup import blend, cold, filtered_blend, warm

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(os.path.dirname(HERE), "Images")
CSV_PATH = os.path.join(IMAGES, "grid_results.csv")
REQ_PATH = os.path.join(IMAGES, "grid_required_range.csv")


POLICIES = ("prepare", "forecast_forward", "defer")
METHOD_KEYS = ("cold", "warm", "blend", "filtered")

FACTORIES = {
    "cold": cold,
    "warm": warm,
    "blend": lambda: blend(0.5),
    "filtered": lambda: filtered_blend(0.5),
}
LABELS = {
    "cold": "cold",
    "warm": "warm",
    "blend": "blend 0.5 s",
    "filtered": "filtered 0.5 s",
}

FIELDS = [
    "policy", "h", "tau", "method", "bearing_deg", "detect_range",
    "captured", "committed", "captured_after_commit", "cert_held_at_commit",
    "false_certificate", "surveying_at_commit", "turn_angle_at_commit",
    "cmd_vel_jump", "cmd_accel_jump", "peak_cmd_accel", "peak_cmd_slew",
    "peak_accel", "min_margin_after_commit", "min_distance", "geofence_breach",
]


def make_scenario(policy, h, tau, method_key, bearing, detect_range):
    return Scenario(
        detect_range=detect_range, bearing=bearing, h=h, policy=policy,
        startup=FACTORIES[method_key](), v_flee=V_FLEE,
        params=Params(lag=True, tau=tau, dt=DT, v_max=V_FLEE, t_eval=T_EVAL),
    )


def case_worker(task):
    """One grid cell. Must be a module-level function so it can be pickled."""
    policy, h, tau, method_key, bearing, detect_range = task
    r = run(make_scenario(policy, h, tau, method_key, bearing, detect_range))
    margin = r.min_margin_after_commit
    return {
        "policy": policy, "h": h, "tau": tau, "method": method_key,
        "bearing_deg": round(math.degrees(bearing), 1),
        "detect_range": round(detect_range, 3),
        "captured": int(r.captured),
        "committed": int(r.committed),
        "captured_after_commit": int(r.captured_after_commit),
        "cert_held_at_commit": int(r.cert_held_at_commit),
        "false_certificate": int(r.false_certificate),
        "surveying_at_commit": int(r.surveying_at_commit),
        "turn_angle_at_commit": round(r.turn_angle_at_commit, 2),
        "cmd_vel_jump": round(r.cmd_vel_jump, 4),
        "cmd_accel_jump": round(r.cmd_accel_jump, 4),
        "peak_cmd_accel": round(r.peak_cmd_accel, 4),
        "peak_cmd_slew": round(r.peak_cmd_slew, 4),
        "peak_accel": round(r.peak_accel, 4),
        # None, not "" -- the csv writer turns None into an empty field, and
        # the in-memory rows then match what load_rows() produces.
        "min_margin_after_commit": (None if math.isnan(margin) else round(margin, 4)),
        "min_distance": round(r.min_distance, 4),
        "geofence_breach": int(r.geofence_breach),
    }


def reqrange_worker(task):
    """Smallest detection range that still escapes, at one bearing."""
    policy, h, tau, method_key, bearing = task
    lo, hi, tol = 1.0, 60.0, 0.25

    def escapes(rng):
        return not run(make_scenario(policy, h, tau, method_key, bearing, rng)).captured

    if not escapes(hi):
        return (policy, h, tau, method_key, None)
    if escapes(lo):
        return (policy, h, tau, method_key, lo)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if escapes(mid):
            hi = mid
        else:
            lo = mid
    return (policy, h, tau, method_key, hi)


def build_tasks(quick: bool):
    bs = bearings(6) if quick else bearings()
    rs = ranges(6, 5.0, 40.0) if quick else ranges()
    hs = (1.0,) if quick else H_VALUES
    taus = (0.3,) if quick else TAU_VALUES

    cases = [
        (policy, h, tau, mk, b, r)
        for policy in POLICIES
        for h in hs
        for tau in taus
        for mk in METHOD_KEYS
        for b in bs
        for r in rs
    ]
    reqs = [
        (policy, h, tau, mk, b)
        for policy in POLICIES
        for h in hs
        for tau in taus
        for mk in METHOD_KEYS
        for b in bs
    ]
    return cases, reqs


def progress(done, total, started, what):
    # The bar redraws with \r, which is noise when stderr is piped to a file.
    if not sys.stderr.isatty():
        return
    if done % max(1, total // 40) and done != total:
        return
    frac = done / total
    elapsed = time.time() - started
    eta = elapsed / frac - elapsed if frac > 0 else 0.0
    bar = "#" * int(frac * 28)
    sys.stderr.write(
        f"\r  {what:<12} [{bar:<28}] {done}/{total}  eta {eta/60:5.1f} min"
    )
    sys.stderr.flush()
    if done == total:
        sys.stderr.write("\n")


def run_grid(quick: bool, workers: int, cases_too: bool = True):
    """`cases_too=False` redoes only the bisections.

    The two halves are independent computations that happen to share a task
    list, so needing one is never a reason to redo the other.
    """
    cases, reqs = build_tasks(quick)
    if cases_too:
        print(f"grid: {len(cases)} cases + {len(reqs)} bisections on {workers} workers\n")
    else:
        print(f"grid: {len(reqs)} bisections only, reusing {CSV_PATH}\n")

    rows = []
    if cases_too:
        started = time.time()
        with Pool(workers) as pool:
            for i, row in enumerate(pool.imap_unordered(case_worker, cases, chunksize=16), 1):
                rows.append(row)
                progress(i, len(cases), started, "cases")
    else:
        rows = load_rows()

    req = {}
    started = time.time()
    with Pool(workers) as pool:
        for i, out in enumerate(pool.imap_unordered(reqrange_worker, reqs, chunksize=4), 1):
            policy, h, tau, mk, value = out
            key = (policy, h, tau, mk)
            prev = req.get(key, -1.0)
            if value is None or prev is None:
                req[key] = None
            else:
                req[key] = max(prev, value)
            progress(i, len(reqs), started, "req range")

    os.makedirs(IMAGES, exist_ok=True)
    if cases_too:
        with open(CSV_PATH, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    with open(REQ_PATH, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "h", "tau", "method", "required_range"])
        for (policy, h, tau, mk), value in sorted(req.items()):
            w.writerow([policy, h, tau, mk, "" if value is None else round(value, 2)])

    if cases_too:
        print(f"\nwrote {len(rows)} rows to {CSV_PATH}")
    print(f"wrote {len(req)} required ranges to {REQ_PATH}")

    return rows, req


def load_req():
    """Required detection ranges are a separate computation from the grid, so
    they get their own file -- otherwise `--tables` re-prints them as n/a."""
    if not os.path.exists(REQ_PATH):
        return {}
    out = {}
    with open(REQ_PATH) as fh:
        for row in csv.DictReader(fh):
            value = row["required_range"]
            out[(row["policy"], float(row["h"]), float(row["tau"]),
                 row["method"])] = None if value == "" else float(value)
    return out


def load_rows():
    with open(CSV_PATH) as fh:
        out = []
        for row in csv.DictReader(fh):
            for k in ("h", "tau", "detect_range", "cmd_vel_jump", "cmd_accel_jump",
                      "peak_cmd_accel", "peak_cmd_slew", "peak_accel",
                      "min_distance", "bearing_deg", "turn_angle_at_commit"):
                row[k] = float(row[k])
            for k in ("captured", "committed", "captured_after_commit",
                      "cert_held_at_commit", "false_certificate",
                      "surveying_at_commit", "geofence_breach"):
                row[k] = int(row[k])
            m = row["min_margin_after_commit"]
            row["min_margin_after_commit"] = None if m == "" else float(m)
            out.append(row)
        return out


def rule(n=96):
    print("-" * n)


def section4_table(rows, req, tau, policy):
    sel = [r for r in rows if r["tau"] == tau and r["policy"] == policy]
    if not sel:
        return

    print(f"\ntau = {tau} s   policy = {policy}   (v_flee = {V_FLEE} m/s, assumed)")
    rule()
    print(f"{'method':<16}{'h':>5}{'cmd dv':>9}{'cmd da':>9}{'min margin':>12}"
          f"{'captures':>10}{'false cert':>12}{'req range':>11}{'cases':>7}")
    print(f"{'':<16}{'(s)':>5}{'(m/s)':>9}{'(m/s^2)':>9}{'(m)':>12}"
          f"{'':>10}{'':>12}{'(m)':>11}{'':>7}")
    rule()

    for mk in METHOD_KEYS:
        for h in sorted({r["h"] for r in sel}):
            cell = [r for r in sel if r["method"] == mk and r["h"] == h]
            if not cell:
                continue
            # Jolt metrics only mean something where the switch happened.
            sw = [r for r in cell if r["committed"]]
            if not sw:
                continue
            margins = [r["min_margin_after_commit"] for r in cell
                       if r["min_margin_after_commit"] is not None]
            rr = req.get((policy, h, tau, mk))
            rr_s = "n/a" if rr is None else f"{rr:.1f}"
            print(
                f"{LABELS[mk]:<16}{h:>5g}"
                f"{max(r['cmd_vel_jump'] for r in sw):>9.2f}"
                f"{max(r['cmd_accel_jump'] for r in sw):>9.1f}"
                f"{(min(margins) if margins else float('nan')):>12.2f}"
                f"{sum(r['captured_after_commit'] for r in cell):>10}"
                f"{sum(r['false_certificate'] for r in cell):>12}"
                f"{rr_s:>11}{len(cell):>7}"
            )
        rule()


def markdown_document(rows, req):
    """The section 4 deliverable, in the handoff's own column order.

    The handoff names `prepare` and `forecast_forward`; `defer` is included
    after them as the always-surveying reference.
    """
    taus = sorted({r["tau"] for r in rows})
    print("# Section 4 results table")
    print()
    print("Generated by `python3 grid.py --tables --markdown` from "
          "`Images/grid_results.csv`.")
    print("One row per start-up method x *h*. 432 detection cases per row ")
    print("(24 bearings x 18 ranges).")
    print()
    print("> **Assumption, not a result:** evader top speed `v_flee = 8 m/s`. "
          "The handoff")
    print("> never states one. Every number below moves if it changes. "
          "See findings_001.md section 4.")
    print()
    print("Columns follow the handoff. `peak jump in commanded acceleration` "
          "is the change")
    print("in *demanded* acceleration `(v_cmd - v)/tau` at commit -- "
          "`d(v_cmd)/dt` at a step")
    print("is unbounded and measures the step size rather than the controller.")
    print()
    print("Jolt columns count only cases that reached the handover; a case "
          "caught before")
    print("*h* never commits.")

    for policy in POLICIES:
        print()
        print(f"## Wait policy: `{policy}`")
        for tau in taus:
            primary = abs(tau - 0.3) < 1e-9
            print()
            print(f"### tau = {tau:g} s" + ("  (primary)" if primary else "  (companion)"))
            print()
            print("| method | h (s) | peak cmd accel jump (m/s2) | peak cmd vel "
                  "jump (m/s) | smallest margin after commit (m) | captures after "
                  "commit | false certificates | required detection range (m) |")
            print("|---|---|---|---|---|---|---|---|")
            for mk in METHOD_KEYS:
                for h in sorted({r["h"] for r in rows}):
                    cell = [r for r in rows if r["policy"] == policy
                            and r["method"] == mk and r["h"] == h
                            and r["tau"] == tau]
                    sw = [r for r in cell if r["committed"]]
                    if not sw:
                        continue
                    margins = [r["min_margin_after_commit"] for r in sw
                               if r["min_margin_after_commit"] is not None]
                    rr = req.get((policy, h, tau, mk))
                    rr_s = "n/a" if rr is None else f"{rr:.1f}"
                    print(f"| {LABELS[mk]} | {h:g} "
                          f"| {max(r['cmd_accel_jump'] for r in sw):.1f} "
                          f"| {max(r['cmd_vel_jump'] for r in sw):.2f} "
                          f"| {(min(margins) if margins else float('nan')):.2f} "
                          f"| {sum(r['captured_after_commit'] for r in cell)} "
                          f"| {sum(r['false_certificate'] for r in cell)} "
                          f"| {rr_s} |")


def surveying_hypothesis(rows):
    """The handoff's claim: the jolt is largest when switching straight from
    surveying.

    Tested against the turn angle at commit rather than a policy label -- how
    far the drone still had to turn is the physical quantity, and it does not
    depend on which controller happened to be active.
    """
    sw = [r for r in rows if r["committed"]]
    print("\n\nTHE JOLT VS HOW FAR THE DRONE STILL HAD TO TURN")
    print("Turn angle at commit: 0 deg = already pointing away, 180 = straight")
    print("at the pursuer. The handoff predicts the jolt grows with this angle.")
    rule(78)
    print(f"{'turn angle at commit':<24}{'cases':>8}{'mean cmd dv':>14}"
          f"{'median':>10}{'max':>9}")
    rule(78)
    bands = [(0, 15), (15, 45), (45, 90), (90, 135), (135, 181)]
    for lo, hi in bands:
        cell = [r for r in sw if lo <= r["turn_angle_at_commit"] < hi]
        if not cell:
            continue
        dv = sorted(r["cmd_vel_jump"] for r in cell)
        mid = dv[len(dv) // 2]
        print(f"{f'{lo}-{hi if hi <= 180 else 180} deg':<24}{len(cell):>8}"
              f"{sum(dv)/len(dv):>14.3f}{mid:>10.3f}{dv[-1]:>9.3f}")
    rule(78)

    print("\nBy wait policy, for reference:")
    rule(78)
    print(f"{'policy':<20}{'cases':>8}{'mean turn angle':>18}{'mean cmd dv':>14}")
    rule(78)
    for policy in POLICIES:
        cell = [r for r in sw if r["policy"] == policy]
        if not cell:
            continue
        print(f"{policy:<20}{len(cell):>8}"
              f"{sum(r['turn_angle_at_commit'] for r in cell)/len(cell):>18.1f}"
              f"{sum(r['cmd_vel_jump'] for r in cell)/len(cell):>14.3f}")
    rule(78)


def extras(rows):
    print("\n\nOTHER THINGS THE GRID SHOWS")
    rule(72)

    total_fc = sum(r["false_certificate"] for r in rows)
    total_cap = sum(r["captured_after_commit"] for r in rows)
    print(f"false certificates across the whole grid: {total_fc} "
          f"(out of {total_cap} captures after commit, {len(rows)} cases)")

    breaches = sum(r["geofence_breach"] for r in rows)
    print(f"geofence breaches: {breaches} of {len(rows)} cases "
          f"({100.0*breaches/len(rows):.0f}%) -- recorded, not prevented")

    ff = [r for r in rows if r["policy"] == "forecast_forward"]
    if ff:
        kept = sum(r["surveying_at_commit"] for r in ff)
        lost = sum(r["captured"] for r in ff if r["surveying_at_commit"])
        print(f"forecast_forward kept surveying in {100.0*kept/len(ff):.0f}% of cases "
              f"and lost {lost} of them")

    print("\nworst-case jolt by method, over the entire grid:")
    for mk in METHOD_KEYS:
        cell = [r for r in rows if r["method"] == mk and r["committed"]]
        if cell:
            print(f"  {LABELS[mk]:<16} cmd dv {max(r['cmd_vel_jump'] for r in cell):6.2f} m/s"
                  f"   cmd da {max(r['cmd_accel_jump'] for r in cell):7.1f} m/s^2")
    rule(72)


def main(argv):
    quick = "--quick" in argv
    tables_only = "--tables" in argv
    req_only = "--reqonly" in argv
    as_markdown = "--markdown" in argv

    if tables_only:
        rows = load_rows()
        req = load_req()
    else:
        workers = max(1, (os.cpu_count() or 4) - 2)
        rows, req = run_grid(quick, workers, cases_too=not req_only)

    if as_markdown:
        markdown_document(rows, req)
        return 0

    print("\n" + "=" * 96)
    print("SECTION 4 RESULTS TABLES")
    print("=" * 96)
    for policy in POLICIES:
        for tau in sorted({r["tau"] for r in rows}):
            section4_table(rows, req, tau, policy)

    surveying_hypothesis(rows)
    extras(rows)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.exit(main(sys.argv[1:]))
