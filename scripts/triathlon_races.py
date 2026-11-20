"""Rider profile from Garmin data, then the three triathlon bike legs:
what was ridden, steady power, and the optimum at equal normalised power.

    .venv/bin/python scripts/triathlon_races.py

Needs data/raw/garmin/csv/*.csv (scripts/garmin_parse.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import versailles_70_3 as V  # noqa: E402
from ttopt import Cyclist, courses, simulate, solve_collocation  # noqa: E402
from ttopt.real import fourth_power_mean, power_on_mesh  # noqa: E402

CSV = ROOT / "data" / "raw" / "garmin" / "csv"

# Fitted on the Thun power file (see the profile section of the output):
# total mass from the energy balance on the climbs, drag area from the flats.
YOU = Cyclist(name="you", mass=70.0, cda=0.29, crr=0.004, cp=235.0, w_prime=18_000.0, p_max=600.0, f_max=2000.0)

RACES = {
    "thun": dict(id=20173577127, title="Ironman Thun 2025, bike leg", window=(5640.0, 29413.0), rho=1.12, power=True),
    "shasta": dict(id=24004018923, title="70.3 Northern California 2026 (Redding), bike leg", window=None, rho=1.13, power=False),
    "versailles": dict(id=24687012359, title="70.3 Versailles 2026, bike leg", window=None, rho=1.23, power=False),
}


def read(activity, window=None):
    a = np.genfromtxt(CSV / f"{activity}.csv", delimiter=",", names=True)
    keep = ~np.isnan(a["lat"]) & ~np.isnan(a["altitude_m"])
    if window:
        keep &= (a["t"] >= window[0]) & (a["t"] <= window[1])
    a = a[keep]
    return a


def limits(v, easy, margin=1.05):
    """Never faster than the recording where the rider was not pedalling."""
    limit = np.full(len(v), v.max() * margin)
    slow = np.concatenate([[False], easy]) | np.concatenate([easy, [False]])
    limit[slow] = v[slow] * margin
    return np.maximum(limit, 4.5)  # a crawl next to a stop is not a corner


def race(key, rider=YOU, scale=None):
    spec = RACES[key]
    a = read(spec["id"], spec["window"])
    # moving time only: stops (aid stations, special needs) are taken off the clock
    step = np.diff(a["t"], prepend=a["t"][0])
    moving = (np.nan_to_num(a["speed_ms"], nan=5.0) >= 1.5) & (step <= 15.0)  # a gap in the file is a stop too
    stopped = float(step[~moving].sum())
    t = np.cumsum(np.where(moving, step, 0.0))
    measured = spec["power"]
    scale = scale or (150.0 if measured else V.SCALE)
    base = courses.from_track(spec["title"], a["lat"], a["lon"], a["altitude_m"], step=20.0, smooth_elevation=80.0,
                              corners=False, rho=spec["rho"])
    flat = np.concatenate([[0.0], np.cumsum(V._haversine(a["lat"][:-1], a["lon"][:-1], a["lat"][1:], a["lon"][1:]))])
    position = np.interp(flat, base.meta["map_distance"], base.distance)
    V.RHO = spec["rho"]
    course = V.at_scale(base, scale)
    ds = 100.0 if course.length < 120_000 else 150.0
    V.DS = ds
    mesh = course.mesh(ds, start_step=None)
    t_nodes, v = V.recorded_speed(mesh, position, t, sigma=scale)
    recorded = t_nodes[-1] - t_nodes[0]
    c = rider
    info = {"hr": float(np.nanmean(a["hr"])), "stopped_s": stopped}

    if measured:
        raw = np.nan_to_num(a["power_w"])
        power, _ = power_on_mesh(mesh, position, raw, t)
        # an interval that contains a stop has almost no recorded power: the replay
        # would stall there. Give it the power of its neighbours instead.
        lost = np.histogram(position[~moving], bins=mesh.s, weights=step[~moving])[0]
        for k in np.flatnonzero(lost > 10.0):
            near = np.r_[power[max(k - 4, 0):max(k - 1, 0)], power[k + 2:k + 5]]
            power[max(k - 1, 0):k + 2] = near.mean()
        easy = power < 30.0
        mesh.vmax[:] = limits(v, easy)
        # drag area that makes the replay of the measured power match the clock
        lo, hi = 0.15, 0.6
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if simulate(c.with_(cda=mid), mesh, power, float(v[0])).time < recorded:
                lo = mid
            else:
                hi = mid
        c = c.with_(cda=0.5 * (lo + hi))
        info["power_1hz_mean"] = float(raw[moving].mean())
        info["np_1hz"] = V.normalised(raw[moving], step[moving])
    else:
        inverse = V.inverse_power(c, mesh, v)
        power = np.maximum(inverse, 0.0)
        easy = inverse < 30.0

    mesh.vmax[:] = limits(v, easy)

    real = simulate(c, mesh, power, float(v[0]), method="you")
    effort = fourth_power_mean(real)
    opt = solve_collocation(c.with_(p_max=1.6 * effort), mesh, v0=float(v[0]), effort_cap=effort)
    steady = simulate(c, mesh, effort, float(v[0]), method="steady")
    return dict(spec=spec, real=real, same_np=opt, steady=steady, effort=effort, recorded=recorded, mesh=mesh, v=v,
                rider=c, info=info, course=course, t_nodes=t_nodes)


def by_gradient(r):
    real, opt, mesh = r["real"], r["same_np"], r["mesh"]
    grade = np.diff(mesh.h) / mesh.ds * 100
    rows = []
    for lo, hi in ((-99, -3), (-3, -1), (-1, 1), (1, 3), (3, 5), (5, 99)):
        pick = (grade >= lo) & (grade < hi)
        if real.dt[pick].sum() < 60:
            continue
        mine = (real.power[pick] * real.dt[pick]).sum() / real.dt[pick].sum()
        best = (opt.power[pick] * opt.dt[pick]).sum() / opt.dt[pick].sum()
        rows.append((lo, hi, mine, best, 100 * real.dt[pick].sum() / real.time))
    return rows


def main():
    print(f"rider: {YOU.mass:.0f} kg with bike, CdA {YOU.cda} m2, Crr {YOU.crr}, CP {YOU.cp:.0f} W, W' {YOU.w_prime / 1000:.0f} kJ")
    for key in RACES:
        r = race(key)
        real, opt, steady, c = r["real"], r["same_np"], r["steady"], r["rider"]
        print(f"\n=== {r['spec']['title']} ===")
        print(f"  {r['course'].length / 1000:.1f} km, {r['course'].total_ascent:.0f} m ascent, recorded {V.clock(r['recorded'])}"
              f" ({3.6 * r['course'].length / r['recorded']:.1f} km/h), mean HR {r['info']['hr']:.0f}, stopped {r['info']['stopped_s']:.0f} s")
        print(f"  power: {'measured' if r['spec']['power'] else 'rebuilt from GPS'}; CdA used {c.cda:.3f}; {r['info']}")
        print(f"  replay error {real.time - r['recorded']:+.0f} s; clock rms {np.sqrt(np.mean((real.t - (r['t_nodes'] - r['t_nodes'][0])) ** 2)):.0f} s")
        for name, plan in (("you", real), ("steady", steady), ("optimal", opt)):
            print(f"  {name:8s} {V.clock(plan.time)}  gain {real.time - plan.time:6.0f} s  mean {plan.mean_power:4.0f} W  p4 {fourth_power_mean(plan):4.0f} W"
                  f"  max {plan.power.max():4.0f} W  {plan.info.get('status', '')}")
        for lo, hi, mine, best, share in by_gradient(r):
            print(f"     gradient {lo:>4} to {hi:<3}: you {mine:4.0f} W  optimal {best:4.0f} W   ({share:4.1f} % of time)")
        V.strategies_figure(
            r, ROOT / "figures" / f"18_{key}_strategies.png",
            title=f"{r['spec']['title']}: three ways to spend the same effort",
            assume=f"{c.mass:.0f} kg with the bike, drag area {c.cda:.2f} m², no wind.",
            rode="What you rode (measured power)" if r["spec"]["power"] else "What you rode (rebuilt from GPS)")


if __name__ == "__main__":
    main()
