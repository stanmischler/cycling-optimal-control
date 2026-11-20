"""Builds time_trial_pacing.ipynb. Run it, then execute the notebook."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(r"""
# Optimal pacing for a cycling time trial

**Question.** Given a rider and a course, how should the rider spread their effort to reach the finish as fast as possible?

**Answer in one call.** `solve(cyclist, course)` returns the power to hold at every point of the course, together with the speed, the elapsed time and the reserve that is left.

This notebook goes through the model, the two solvers (direct collocation and dynamic programming), how they compare, and what they say about two real Tour de France time trials.
""")

md(r"""
## 1. The model

**The bike.** Along the road, with distance $s$ as the independent variable, speed $v$, total mass $m$ and rider power $P$:

$$ m\,v\,\frac{dv}{ds} \;=\; \frac{\eta P}{v} \;-\; \tfrac12 \rho\, C_dA\,(v+w)^2 \;-\; C_{rr}\, m g \cos\theta \;-\; m g \sin\theta \;-\; B $$

The terms are drive force, aerodynamic drag (with headwind $w$), rolling resistance, gravity and braking $B \ge 0$.

**The rider.** Two numbers describe what the rider can do:

- critical power $CP$: sustainable for the whole race;
- a reserve $W'$ (joules): every watt above $CP$ drains it, and **it never refills**.

$$ \frac{dW}{dt} = -\max(P - CP,\, 0), \qquad W \ge 0, \qquad 0 \le P \le P_{max} $$

**The goal.** Minimise the finish time $T = \int_0^L \frac{ds}{v}$.

**Discretisation.** The course is cut into intervals. On each one we apply the work-energy theorem exactly, so gravity is integrated without error and energy is conserved by construction. The same discrete model is used by both solvers and by the simulator, so their results can be compared directly.
""")

code(r"""
import json, warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from ttopt import Cyclist, courses, solve, simulate, viz
from ttopt.study import refinement
from ttopt.races import load_race
from ttopt.real import compare

rider = Cyclist()          # a generic professional
rider
""")

md(r"""
## 2. First example: one hill

A 10 km course with 2 km at 7 % in the middle. Three plans are compared:

- **Collocation** and **dynamic programming**: the two optimisers;
- **Constant power**: the natural baseline, the single power that empties the reserve exactly on the line.
""")

code(r"""
hill = courses.single_hill()
mesh = hill.mesh(25)
best = solve(rider, mesh, "collocation")
dp = solve(rider, mesh, "dp")
flat_out = solve(rider, mesh, "constant")
for plan in (best, dp, flat_out):
    print(plan.summary())
viz.plot_plan(best, [dp, flat_out], labels=["Collocation", "Dynamic programming", "Constant power"],
              title="Single hill: 10 km with 2 km at 7 %");
""")

md(r"""
**Reading the result.** Power up to $CP$ costs nothing, so the rider never goes below it. The reserve is worth most where speed is lowest, because that is where a watt buys the most seconds: all of it goes into the climb. The two optimisers agree, and both beat constant power by about 15 s.

The solver also returns the *price* of the reserve: how many seconds one more kilojoule would buy.
""")

code(r"""
print(f"One extra kJ of reserve is worth {best.info['reserve_price_s_per_kJ']:.2f} s on this course")
print(best.check())
""")

md(r"""
## 3. Four synthetic courses

The flat course shows the textbook structure: maximum power off the line, then constant speed, then back to $CP$ once the reserve is gone.
""")

code(r"""
titles = {"flat": "Flat, 10 km", "single hill": "Single hill", "rolling": "Rolling: 20 km of 80 m waves",
          "climb and descent": "Climb and descent: 12 km at 6 %"}
print(f"{'course':<20s}{'collocation':>13s}{'dyn. prog.':>13s}{'constant':>13s}{'gain vs constant':>19s}")
plans = {}
for name, make in courses.SYNTHETIC.items():
    m = make().mesh(25)
    a, b, c = solve(rider, m), solve(rider, m, "dp"), solve(rider, m, "constant")
    plans[name] = (a, b, c)
    print(f"{name:<20s}{viz.fmt_time(a.time):>13s}{viz.fmt_time(b.time):>13s}{viz.fmt_time(c.time):>13s}{c.time - a.time:>17.1f} s")
""")

code(r"""
a, b, c = plans["flat"]
viz.plot_plan(a, [b, c], labels=["Collocation", "Dynamic programming", "Constant power"], title=titles["flat"]);
""")

code(r"""
a, b, c = plans["rolling"]
viz.plot_plan(a, [b, c], labels=["Collocation", "Dynamic programming", "Constant power"], title=titles["rolling"]);
""")

md(r"""
## 4. The two solvers

**Direct collocation.** Every speed and every force along the course is an unknown; the physics become equality constraints; one large sparse optimisation problem is handed to IPOPT. Because the reserve never refills, its whole history collapses into one budget constraint.

**Dynamic programming.** Work backwards from the finish. At each point of the course, for every speed and every reserve level on a grid, store the best achievable time to the finish. The ride is then read off forwards. It is slower, but it gives more: the best action from *any* state, not just along one ride.

Both are run at increasing resolution and compared with a very fine collocation run.
""")

code(r"""
rows, reference = refinement(rider, hill, verbose=False)
print(f"reference finish time: {reference.time:.4f} s on {reference.mesh.n} intervals\n")
print(f"{'method':<13s}{'settings':<40s}{'error (s)':>11s}{'solve time (s)':>16s}")
for r in rows:
    settings = ", ".join(f"{k}={r[k]:g}" for k in ("ds", "dv", "n_w") if k in r)
    print(f"{r['method']:<13s}{settings:<40s}{r['error_s']:>+11.4f}{r['seconds']:>16.3f}")
viz.plot_refinement(rows, "Single hill: accuracy against computing time");
""")

md(r"""
**What this shows.**

- Both solvers converge to the same finish time, which is the main cross-check of the whole project.
- Collocation is second-order accurate in the interval length and reaches a hundredth of a second in about 0.1 s.
- Dynamic programming needs roughly ten times longer for the same accuracy, and its error does not shrink smoothly, because three grids (distance, speed, reserve) interact.
- Everything runs in seconds on a laptop. A supercomputer is not needed for this problem.
""")

md(r"""
## 5. Two real courses

- **Monaco → Nice**, Tour de France 2024 stage 21, 33.7 km: La Turbie, Col d'Èze, then a long descent.
- **Passy → Combloux**, Tour de France 2023 stage 16, 22.4 km: rolling, then the Côte de Domancy and a rising finish.

The tracks are public route files with elevation from the French national terrain model. Corner speed limits are derived from the curvature of the track.
""")

code(r"""
races = {}
for key in ("monaco_nice_2024", "passy_combloux_2023"):
    course, checkpoints, times, labels = load_race(key)
    races[key] = (course, checkpoints, times, labels)
    print(f"{course.name}: {course.length/1000:.2f} km, {course.total_ascent:.0f} m of climbing, "
          f"steepest {100*np.gradient(course.elevation, course.distance).max():.0f} %")
""")

code(r"""
course, checkpoints, times, labels = races["monaco_nice_2024"]
nice = solve(rider, course.mesh(50, breaks=checkpoints[:-1]))
nice_dp = solve(rider, nice.mesh, "dp")
nice_const = solve(rider, nice.mesh, "constant")
for plan in (nice, nice_dp, nice_const):
    print(plan.summary())
viz.plot_plan(nice, [nice_const], labels=["Optimal", "Constant power"], checkpoints=checkpoints,
              checkpoint_labels=labels, smooth_m=300, title="Monaco → Nice: optimal plan for the generic rider");
""")

code(r"""
viz.plot_map(nice, course, title="Monaco → Nice: where the power goes");
""")

code(r"""
course, checkpoints, times, labels = races["passy_combloux_2023"]
combloux = solve(rider, course.mesh(50, breaks=checkpoints[:-1]))
combloux_dp = solve(rider, combloux.mesh, "dp")
combloux_const = solve(rider, combloux.mesh, "constant")
for plan in (combloux, combloux_dp, combloux_const):
    print(plan.summary())
viz.plot_plan(combloux, [combloux_const], labels=["Optimal", "Constant power"], checkpoints=checkpoints,
              checkpoint_labels=labels, smooth_m=300, title="Passy → Combloux: optimal plan for the generic rider");
""")

code(r"""
viz.plot_map(combloux, course, title="Passy → Combloux: where the power goes");
""")

md(r"""
## 6. Against real riders

**What data exists.** No rider's full power file is public for either race. What we have:

- official times at three checkpoints and the finish, for every rider;
- measured average power on single climbs for two riders (from their own uploads, quoted in the press): Derek Gee on La Turbie (446 W for 17:46) and the Col d'Èze (483 W for 4:41), Neilson Powless on the Côte de Domancy (430 W for 7:20).

**Step 1: does the model reproduce the measured climbs?** We ask the model what power it takes to ride each climb in the measured time.
""")

code(r"""
from ttopt.model import Course
from ttopt.races import map_km_to_road

def power_for(course, km0, km1, seconds, rider, v_in):
    a, b = map_km_to_road(course, km0), map_km_to_road(course, km1)
    keep = (course.distance >= a) & (course.distance <= b)
    part = Course("part", course.distance[keep] - course.distance[keep][0], course.elevation[keep],
                  v_limit=course.v_limit[keep], rho=course.rho).mesh(20, start_step=None)
    lo, hi = 100.0, 900.0
    for _ in range(40):
        p = 0.5 * (lo + hi)
        lo, hi = (p, hi) if simulate(rider.with_(p_max=2000), part, p, v_in).time > seconds else (lo, p)
    return p

gee = Cyclist(name="Gee", mass=81, cda=0.22)            # 71 kg rider + 10 kg of bike and kit
powless = Cyclist(name="Powless", mass=76, cda=0.28, crr=0.0035)   # he rode a road bike
nice_course, combloux_course = races["monaco_nice_2024"][0], races["passy_combloux_2023"][0]
checks = [("Gee, La Turbie", power_for(nice_course, 3.2, 11.2, 1066, gee, 10.0), 446),
          ("Gee, Col d'Èze", power_for(nice_course, 15.55, 17.15, 281, gee, 9.0), 483),
          ("Powless, Côte de Domancy", power_for(combloux_course, 16.5, 18.9, 440, powless, 11.0), 430)]
print(f"{'climb':<28s}{'model (W)':>10s}{'measured (W)':>14s}{'difference':>12s}")
for name, model, measured in checks:
    print(f"{name:<28s}{model:>10.0f}{measured:>14.0f}{100*(model/measured-1):>+11.1f} %")
""")

md(r"""
The model reads about 5 % low on all three climbs. The bias is the same everywhere, so *relative* effort between sections, which is what pacing is about, is reproduced to about 1 %. The exact position of each measured section on our track is not known, which accounts for part of the gap.

**Step 2: rebuild the real ride from the split times.** Four times do not define a ride, so we take the most favourable reading: the rider is assumed to pace *perfectly inside* each segment, and only the split of effort *between* segments is taken from the clock. Their critical power is the lowest one that makes those times possible. The optimiser then gets the same rider on the same course. The difference is a cautious estimate of what better pacing was worth.
""")

code(r"""
course, checkpoints, times, labels = races["passy_combloux_2023"]
vingegaard = Cyclist(name="Vingegaard", mass=70, cda=0.20, p_max=1000)
duel = compare(vingegaard, course, checkpoints, times["Jonas Vingegaard"])
print(f"fitted critical power {duel.cyclist.cp:.0f} W, estimated gain {duel.gain_s:.1f} s\n")
print(f"{'segment':<26s}{'real time':>11s}{'optimal':>10s}{'real W':>9s}{'optimal W':>11s}")
for name, r in zip(labels, duel.segment_table()):
    print(f"{'to ' + name:<26s}{r['real_time_s']:>10.0f}s{r['optimal_time_s']:>9.0f}s{r['real_power_W']:>9.0f}{r['optimal_power_W']:>11.0f}")
viz.plot_plan(duel.optimal, [duel.real], labels=["Optimal", "Vingegaard, rebuilt from his split times"],
              checkpoints=checkpoints, checkpoint_labels=labels, smooth_m=300,
              title="Passy → Combloux 2023: Vingegaard against the optimal plan");
""")

code(r"""
course, checkpoints, times, labels = races["monaco_nice_2024"]
duel = compare(gee.with_(p_max=1000), course, checkpoints, times["Derek Gee"])
print(f"fitted critical power {duel.cyclist.cp:.0f} W, estimated gain {duel.gain_s:.1f} s\n")
print(f"{'segment':<26s}{'real time':>11s}{'optimal':>10s}{'real W':>9s}{'optimal W':>11s}")
for name, r in zip(labels, duel.segment_table()):
    print(f"{'to ' + name:<26s}{r['real_time_s']:>10.0f}s{r['optimal_time_s']:>9.0f}s{r['real_power_W']:>9.0f}{r['optimal_power_W']:>11.0f}")
viz.plot_plan(duel.optimal, [duel.real], labels=["Optimal", "Gee, rebuilt from his split times"],
              checkpoints=checkpoints, checkpoint_labels=labels, smooth_m=300,
              title="Monaco → Nice 2024: Gee against the optimal plan");
""")

md(r"""
**Step 3: how solid is the number?** The estimate rests on things we do not know: the rider's drag area and how fast they can corner. The scan below recomputes the gain for each rider over a grid of both (`scripts/gain_scan.py`).
""")

code(r"""
scan = json.loads(open("data/gain_scan.json").read())
names = {"monaco_nice_2024": "Monaco → Nice 2024", "passy_combloux_2023": "Passy → Combloux 2023"}
print(f"{'race':<24s}{'rider':<20s}{'lowest':>8s}{'median':>8s}{'highest':>9s}   (seconds, over the grid)")
for race in names:
    for who in dict.fromkeys(r["rider"] for r in scan if r["race"] == race):
        g = np.array([r["gain_s"] for r in scan if r["race"] == race and r["rider"] == who and r.get("gain_s") is not None])
        print(f"{names[race]:<24s}{who:<20s}{g.min():>8.0f}{np.median(g):>8.0f}{g.max():>9.0f}")
viz.plot_gain_scan(scan, names);
""")

md(r"""
**What the scan says.**

- For each rider the gain is smallest at one drag area and grows on both sides. The lowest value is the most favourable reading of that rider's splits, so it is a floor, not a best guess.
- **The two stage winners are the closest to the optimum**: at least 5 s for Vingegaard at Combloux and at least 7 s for him in Nice.
- The other leaders come out at 14 s or more (Gee), 18 to 20 s (Pogačar, whose Combloux figure includes his bike change), 29 s (Evenepoel) and 33 s (van Aert).
- Powless is the outlier at 149 s or more. His splits are very uneven, which fits a rider who was not racing for the result, so this is not a pacing error.
- Away from the best-fitting drag area the gain climbs past a minute, but the fitted critical power also becomes unrealistic (above 7.5 W/kg), so those corners of the grid are not credible.
- A common pattern in Nice: the optimal plan keeps almost the whole reserve for the Col d'Èze, the last steep climb, where every rider's rebuilt ride spends it earlier on La Turbie. That follows directly from a reserve that never refills.

**Bottom line.** From four split times, the gain for the leading riders is somewhere between a few seconds and about half a minute, and the data cannot narrow it further.
""")

md(r"""
## 7. A real power file: Hepburn, Tour de France 2018 stage 20

One pro time trial does have a full public recording: **Michael Hepburn** (Mitchelton-Scott) on the 31 km hilly time trial from Saint-Pée-sur-Nivelle to Espelette, 28 July 2018. He finished 10th in 42:15. His head-unit file (power, speed and GPS every second) was shared publicly through a TrainingPeaks race analysis. The data is for personal analysis only and is not redistributed here.

**Method.**

1. The course is built from his own GPS track, with altitude from the French terrain model.
2. His recorded power is fed to the model. One number, the drag area, is tuned so the model finishes in his real time.
3. Wherever he stopped pedalling (corners, fast descents), his real speed becomes a speed limit, so the optimiser can never out-corner him.
4. The optimiser plans the same rider's best ride.
""")

code(r"""
import sys
sys.path.insert(0, "scripts")
import hepburn_2018 as hepburn

out = hepburn.three_way()
real, fitted, err = out["real"], out["rider"], out["clock_error"]
print(f"course: {out['course'].length/1000:.2f} km, {out['course'].total_ascent:.0f} m of climbing")
print(f"recorded power: mean {out['data']['power_w'].mean():.0f} W, normalised {out['np_real']:.0f} W")
print(f"fitted drag area: {fitted.cda:.3f} m²")
print(f"model clock against his real clock along the course: rms {np.sqrt((err**2).mean()):.1f} s, worst {np.abs(err).max():.1f} s")
viz.plot_replay_check(real, out["position"], out["speed"], err);
""")

md(r"""
**The model holds.** Fed his real power, it stays within about 5 s of his real clock over 42 minutes (0.2 %), and never drifts more than 11 s.

**What does "same effort" mean?** This is the crux. His real power is very spiky: he is above his threshold more than half the time and coasts in every corner. Three ways of holding the optimiser to the same effort give three answers.
""")

code(r"""
print(f"real ride: {viz.fmt_time(real.time)}\n")
print(f"{'rule for the optimiser':<34s}{'time':>9s}{'gain':>9s}{'mean W':>8s}{'normalised W':>14s}")
for name, plan in out["plans"].items():
    resampled = np.interp(np.arange(0, plan.time), plan.t[:-1], plan.power)
    print(f"{name:<34s}{viz.fmt_time(plan.time):>9s}{real.time - plan.time:>+8.1f}s{plan.mean_power:>8.0f}{hepburn.normalised_power(resampled):>14.0f}")
""")

md(r"""
- **No refill** (the rule used everywhere else in this notebook): the optimum is *slower* than what he really did. A reserve that never refills is stricter than a real body, which recovers between efforts. His recording is impossible under that rule unless his threshold were about 520 W.
- **Reserve refills** (Skiba model, the rule his recording is read with): the optimiser exploits it, riding at a normalised power of 500 W against his 436 W. No rider with his threshold could do that, so this is an upper bound and not a credible plan.
- **Equal normalised power**: the optimum must not exceed his own normalised power. This is the fairest of the three, and it gives a gain of about **20 s (0.8 %)**.
""")

code(r"""
best = out["plans"]["equal normalised power"]
grade = real.mesh.grade
print(f"{'terrain':<28s}{'real':>9s}{'optimal':>9s}{'saved':>8s}{'real W':>8s}{'optimal W':>11s}")
for lo, hi, label in ((-1, -0.03, "descents steeper than 3 %"), (-0.03, 0.03, "flat to rolling"), (0.03, 1, "climbs steeper than 3 %")):
    m = (grade > lo) & (grade <= hi)
    tr, to = real.dt[m].sum(), best.dt[m].sum()
    print(f"{label:<28s}{tr:>8.0f}s{to:>8.0f}s{tr - to:>+7.0f}s{(real.power[m]*real.dt[m]).sum()/tr:>8.0f}{(best.power[m]*best.dt[m]).sum()/to:>11.0f}")
viz.plot_plan(best, [real], labels=["Optimal at equal normalised power", "Hepburn, recorded power"], smooth_m=200,
              title="Tour de France 2018 stage 20: Hepburn against the optimal plan");
""")

md(r"""
**Reading it.** The optimal plan gains its time on the climbs and gives a little back on the flats and descents. It rides the climbs at almost the same average power as he did, but more evenly and with the hardest efforts on the steepest ramps, and it pedals less on the descents. The bottom panel shows only his reserve (read with the refilling model); the equal-effort plan has no reserve to show.

**How much to trust it.** The 20 s is what this model says is available at equal normalised power. The bracket from the other two rules, from 8 s slower to 108 s faster, shows that the answer is set more by how fatigue is modelled than by the optimiser.
""")

md(r"""
## 8. Limits

- **The reserve never refills** in the main solver. That is the modelling choice made for this project. Section 7 shows its cost: a real pro's ride is not even feasible under it. `solve(..., recovery=True)` lets it refill, and `solve(..., effort_cap=...)` limits normalised power instead; dynamic programming supports only the no-refill rule.
- **Maximum power does not fade** as the reserve empties.
- **Corners** are speed limits derived from a drawn route line, with a floor of 29 km/h; real racing lines are not known.
- **Wind** is supported by the model but set to zero here; the wind on the day was not measured on the course.
- **Real-rider comparison from split times** (section 6): only four times per rider, so the gain is an estimate with a wide range.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, ROOT / "time_trial_pacing.ipynb")
print("wrote", ROOT / "time_trial_pacing.ipynb")
