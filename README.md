# Pursuit/evader simulator

A 2-D toy for the project 2 question: when a drone switches to an escape
controller mid-flight, how big is the jolt and what safety margin does it cost?

Pure Python standard library, no dependencies. Python 3.8+.

## Running it

```sh
python3 run_tests.py          # all 146 tests, ~30 s
python3 run_tests.py -v       # verbose
python3 run_tests.py sim      # one module (tests/test_sim.py)

python3 grid.py               # the full 46,656-case sweep (~13 min)
python3 grid.py --tables      # re-print its tables, no re-running
python3 report.py repro       # experiment 002 comparison (legacy plant)
python3 report.py filter      # where the safety filter overrides the blend
```

## Figures

Two routes, from the same runs and the same colour assignment, so a figure in
the paper and a screenshot of the viewer agree on which colour means what.

```sh
python3 plot.py               # static PNG + PDF -> ../Images/  (needs matplotlib)
python3 export_viz.py         # data for the interactive viewer
```

`plot.py` writes `fig_switch_speed`, `fig_switch_panels` (the two-week
milestone figure: commanded and actual velocity across the switch, head-on at
h = 1 s, all four methods), `fig_margin`, `fig_tracks` and `fig_tau_sweep`.

`viewer/index.html` is a self-contained replay: pick an approach, a detection
range and a start-up method, then scrub the handover. The plan view, all three
strip charts and the metric tiles share one cursor. "Compare all" overlays the
four methods. It needs `viewer/runs.js`, which `export_viz.py` regenerates;
open the file directly in a browser or serve the directory.

Note that matplotlib is only needed for `plot.py`. The simulator and
`run_tests.py` stay standard-library only.

## Layout

| file | what it holds |
|---|---|
| `pursuit_evader/vec.py` | 2-D vector type |
| `pursuit_evader/params.py` | every physical constant, one dataclass |
| `pursuit_evader/dynamics.py` | evader plant (speed loop + caps), pursuer |
| `pursuit_evader/certificate.py` | the margin rule and the delayed estimator |
| `pursuit_evader/controllers.py` | surveillance, hover, prepare, escape |
| `pursuit_evader/startup.py` | cold / warm / blend / filtered, and the safety filter |
| `pursuit_evader/sim.py` | the run loop, `Scenario` and `Result` |
| `pursuit_evader/scenarios.py` | named cases and the 24x18 grid |
| `pursuit_evader/sweep.py` | required detection range, grid summaries |

## The model

**Evader.** A double integrator driven by a *velocity* command:

```
lag on    dv/dt = (v_cmd - v) / tau
lag off   v     = v_cmd
```

Acceleration is capped at `a_max` in both modes, so "lag off" is exactly the
experiment 002 plant: ask to flee, get full acceleration on the same step.
Integration is semi-implicit Euler at a fixed `dt`.

**Pursuer.** Constant speed, pure pursuit, instantaneous turn. Worst case for
the evader, and what the certificate is written against.

**Certificate.**

```
margin = d - rho - r_capture - max(0, v_p - v_r)^2 / (2 a_max) - extra
```

`extra` is the slot for the start-up cost derived in step 8. It is 0 today.

**Timeline.** `t = 0` is detection. The escape controller may not take charge
until `t = h`, the *commit*. Every run is scored over the same window
`[0, t_eval]`, which is what closes the audit's complaint that policies had
been compared over different windows.

## Metrics, and why they are defined this way

| field | definition |
|---|---|
| `cmd_vel_jump` | discontinuity in the velocity command at commit, m/s |
| `cmd_accel_jump` | change in *demanded* acceleration `(v_cmd - v)/tau` at commit |
| `peak_cmd_accel` | largest demanded acceleration anywhere after commit |
| `peak_cmd_slew` | max `d(v_cmd)/dt` strictly after the commit step |
| `peak_accel` | largest *realised* acceleration; never exceeds `a_max` |
| `min_margin_after_commit` | smallest certificate margin after commit |
| `false_certificate` | the rule certified at commit and the drone was caught |

The handoff asks for "peak jump in commanded acceleration". Since the command
is a velocity, `d(v_cmd)/dt` at a handover step is just `dv/dt` and diverges as
`dt` shrinks, so it measures the step size rather than the controller. The
demanded acceleration `(v_cmd - v)/tau` is what the speed loop is actually
asked to produce, converges as `dt` shrinks, and is what `cmd_accel_jump`
reports. `peak_cmd_slew` excludes the commit step so it is well defined for the
blends, where it is the quantity that separates one blend length from another.

## Two parameters the handoff does not pin down

Both are live questions, not settled choices. See `Notes/findings_001.md`.

**`Params.v_max` and `Scenario.v_flee` — the evader's top speed.** Default is
`None`, meaning uncapped, which is what experiment 002 did. That default makes
the escape command a huge saturating velocity, which pins the realised
acceleration at `a_max` at every `tau` — so the speed loop has no effect on
escape authority and steps 5-8 have nothing to measure. Any run studying the
lag must set a finite `v_flee`. `report.py` uses 8.0 m/s and says so.

**`EscapeController.tau_cmd` — the controller's own output smoother.** This is
the "memory" that cold start zeroes and warm start seeds, so the size of the
cold-start jolt scales with it. Default 0.2 s, chosen to be the same order as
`tau`. Nothing in the handoff fixes it.

## Reproduction status

`tests/test_reproduce_002.py` pins these against experiment 002.

| | this build | reported 002 |
|---|---|---|
| prepare, required range at h = 0.5/1/2 | 11.1 / 11.1 / 12.1 m | 11 / 11 / 11 m |
| defer, required range at h = 0.5/1/2 | 15.1 / 19.0 / 26.8 m | 15 / 20 / 28 m |
| stop-first vs prepare, from behind | 1.4x to 2.0x | 1.9x to 4.3x |

The first two reproduce. The third does not: the same direction, a smaller
effect. The likely cause is the escape controller's gains, which the handoff
does not specify.

The 11 m figure is the worst case over all bearings, and the worst bearing is
head-on, so the numeric checks use head-on. Approached from behind, fleeing
straight ahead is what surveying already does, so `defer` and `prepare`
coincide there and both need far less warning than the 11 m headline.
