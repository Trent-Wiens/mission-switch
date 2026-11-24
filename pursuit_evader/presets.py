"""The parameter sets the scripts run against, in one place.

These were copied into five separate files, which meant the number the whole
project rests on could be changed in one script and silently disagree with the
others.  Everything imports them from here now.

Two plants, deliberately distinct:

    LEGACY     what experiment 002 ran -- no speed loop, no speed cap, so the
               escape controller has full immediate authority.  The ONLY fair
               configuration for reproducing 002's numbers.

    REALISTIC  the speed loop modelled, and a finite top speed.  Everything
               after step 3 of the handoff uses this.

Mixing them is how you get a comparison that changes two things at once.
"""

from __future__ import annotations

from .params import Params

#: OPEN QUESTION -- the evader's top speed.  The handoff never states one.
#:
#: Uncapped, the escape command has to be an effectively infinite velocity,
#: which pins the acceleration at a_max regardless of tau -- so the speed loop
#: does nothing and the lag study has no effect to measure.  Capped below the
#: pursuer's 5 m/s, which the handoff's prose implies, the drone can never
#: escape and "safe forever" is false by construction.
#:
#: 8 m/s splits the difference: above the pursuer so escape is possible, finite
#: so the lag bites.  It is an assumption, not a result, and findings_003.md
#: shows it is the dominant driver of the false-certificate count.
V_FLEE = 8.0

#: Seconds of simulated time scored per run, measured from detection.  One
#: window for every policy -- the audit's second caveat was that policies had
#: been compared over different windows.
T_EVAL = 25.0

#: Integration step.  Params rejects dt > tau/4, so this has to be small
#: enough for the shortest tau swept (0.1 s).
DT = 0.005

#: Experiment 002's plant: full immediate authority, no speed cap.
LEGACY = Params(lag=False, v_max=None, dt=DT, t_eval=T_EVAL)

#: The plant everything after step 3 uses.
REALISTIC = Params(lag=True, tau=0.3, v_max=V_FLEE, dt=DT, t_eval=T_EVAL)

#: Values the handoff asks to sweep.
H_VALUES = (0.5, 1.0, 2.0)
TAU_VALUES = (0.1, 0.3, 0.5)
BLEND_TIME = 0.5
