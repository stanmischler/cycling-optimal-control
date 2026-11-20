"""Half Ironman Versailles, bike leg: estimate power from a GPS recording
(no power meter), then compare with the optimal pacing at equal effort.

    .venv/bin/python scripts/versailles_70_3.py

The recording has position, altitude and time only. Power is rebuilt by
inverting the bike model, so every watt below depends on the assumed mass,
drag area and rolling resistance; the script prints how much.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ttopt import Cyclist, courses, simulate, solve_collocation  # noqa: E402
from ttopt.courses import _gaussian_smooth, _haversine  # noqa: E402
from ttopt.model import required_wheel_force  # noqa: E402
from ttopt.real import critical_power_with_recovery, fourth_power_mean  # noqa: E402

GPX = ROOT / "data" / "raw" / "versailles" / "70_3_Versailles_Bike.gpx"
RHO = 1.23  # about 12 C at 120 m, 11 October morning
DS = 100.0

# Assumptions (not measured): total mass, drag area, rolling resistance.
BASE = Cyclist(name="you (assumed)", mass=80.0, cda=0.30, crr=0.004, cp=200.0, w_prime=20_000.0,
               p_max=600.0, f_max=2000.0)


def load():
    import gpxpy

    with open(GPX, encoding="utf-8") as fh:
        gpx = gpxpy.parse(fh)
    pts = [p for trk in gpx.tracks for seg in trk.segments for p in seg.points]
    lat = np.array([p.latitude for p in pts])
    lon = np.array([p.longitude for p in pts])
    ele = np.array([p.elevation for p in pts])
    t = np.array([(p.time - pts[0].time).total_seconds() for p in pts])
    course = courses.from_track("70.3 Versailles, bike", lat, lon, ele, step=20.0, smooth_elevation=80.0,
                                corners=False, rho=RHO)
    flat = np.concatenate([[0.0], np.cumsum(_haversine(lat[:-1], lon[:-1], lat[1:], lon[1:]))])
    position = np.interp(flat, course.meta["map_distance"], course.distance)  # road metres
    return course, position, t


def at_scale(course, scale, wind=(0.0, 0.0)):
    """The course with altitude smoothed over ``scale`` metres, and a steady wind.

    ``wind`` = (speed in m/s, direction it blows from in degrees).
    """
    h = _gaussian_smooth(course.elevation, 20.0, max(scale - 80.0, 1.0))
    lat, lon = np.radians(course.lat), np.radians(course.lon)
    k = np.ones(15) / 15.0
    east = np.convolve(np.gradient(lon) * np.cos(lat), k, "same")
    north = np.convolve(np.gradient(lat), k, "same")
    head = wind[0] * np.cos(np.arctan2(east, north) - np.radians(wind[1]))
    return courses.Course(course.name, course.distance, h, headwind=head, rho=RHO, lat=course.lat, lon=course.lon)


def recorded_speed(mesh, position, t, sigma=150.0):
    """Clock and speed at the mesh nodes, from the recording."""
    keep = np.concatenate([[True], np.diff(position) > 0])
    t_nodes = np.interp(mesh.s, position[keep], t[keep])
    pace = np.gradient(t_nodes, mesh.s)  # s/m
    pace = _gaussian_smooth(pace, DS, sigma)  # GPS jitter; keeps the total time
    return t_nodes, 1.0 / pace


def inverse_power(c: Cyclist, mesh, v):
    """Power the model needs to ride the recorded speed; negative = braking."""
    f = required_wheel_force(c, mesh, v)
    return f * 0.5 * (v[:-1] + v[1:]) / c.efficiency


def rolling_mean_max(power, dt, window):
    """Best average power over ``window`` seconds."""
    t = np.concatenate([[0.0], np.cumsum(dt)])
    e = np.concatenate([[0.0], np.cumsum(power * dt)])
    grid = np.arange(0.0, t[-1] - window, 5.0)
    return float(((np.interp(grid + window, t, e) - np.interp(grid, t, e)) / window).max())


def normalised(power, dt):
    """Normalised power: 30 s rolling mean, then fourth-power mean."""
    t = np.concatenate([[0.0], np.cumsum(dt)])
    e = np.concatenate([[0.0], np.cumsum(power * dt)])
    grid = np.arange(30.0, t[-1], 1.0)
    roll = (np.interp(grid, t, e) - np.interp(grid - 30.0, t, e)) / 30.0
    return float(np.mean(roll**4) ** 0.25)


SCALE = 600.0  # metres: speed and altitude are trusted only above this scale


def analyse(c: Cyclist, course, position, t, optimise=True, scale=SCALE, wind=(0.0, 0.0)):
    course = at_scale(course, scale, wind)
    mesh = course.mesh(DS, start_step=None)
    t_nodes, v = recorded_speed(mesh, position, t, sigma=scale)
    raw = inverse_power(c, mesh, v)
    power = np.maximum(raw, 0.0)
    dt = np.diff(t_nodes)
    out = {
        "mean": float((power * dt).sum() / dt.sum()),
        "np": normalised(power, dt),
        "best20": rolling_mean_max(power, dt, 1200.0),
        "best60": rolling_mean_max(power, dt, 3600.0),
        "braking_share": float(dt[raw < -20.0].sum() / dt.sum()),
        "cp_recovery": critical_power_with_recovery(power, dt, c.w_prime),
        "p4": float(((power**4 * dt).sum() / dt.sum()) ** 0.25),
        "ascent": course.total_ascent,
    }
    if not optimise:
        return out

    # speed limits: never faster than the recording where the rider was not pedalling
    limit = np.full(mesh.n + 1, v.max() * 1.02)
    easy = np.concatenate([[False], raw < 30.0]) | np.concatenate([raw < 30.0, [False]])
    limit[easy] = v[easy] * 1.02
    mesh.vmax[:] = limit

    real = simulate(c, mesh, power, float(v[0]), method="you (rebuilt)")
    effort = fourth_power_mean(real)
    same_np = solve_collocation(c.with_(p_max=1.6 * effort), mesh, v0=float(v[0]), effort_cap=effort)
    # same total work: sustainable power such that CP * T + W' = work done
    work = float((real.power * real.dt).sum())
    cp_work = (work - c.w_prime) / real.time
    same_work = solve_collocation(c.with_(cp=cp_work, p_max=1.6 * effort), mesh, v0=float(v[0]))
    steady = simulate(c, mesh, out["mean"], float(v[0]), method="steady")
    out.update(real=real, same_np=same_np, same_work=same_work, steady=steady, effort=effort,
               recorded=t_nodes[-1] - t_nodes[0], mesh=mesh, v=v)
    return out


def clock(sec):
    sec = int(round(sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def figure(res, path):
    import matplotlib.pyplot as plt

    from ttopt import viz

    viz.style()
    real, opt, mesh = res["real"], res["same_np"], res["mesh"]
    km = mesh.s / 1000.0
    mid = 0.5 * (km[:-1] + km[1:])
    smooth = lambda y: _gaussian_smooth(np.asarray(y, float), DS, 300.0)
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True, gridspec_kw={"height_ratios": [1, 2, 1.2]})
    ax[0].fill_between(km, mesh.h, mesh.h.min() - 5, color="#8a9098", alpha=0.25, lw=0)
    ax[0].plot(km, mesh.h, color="#555b63", lw=1.2)
    ax[0].set_ylabel("Elevation (m)")
    ax[1].plot(mid, smooth(real.power), color="#eb6834", lw=1.6, label="What you rode (rebuilt from GPS)")
    ax[1].plot(mid, smooth(opt.power), color="#2a78d6", lw=1.8, label="Optimal at equal normalised power")
    ax[1].set_ylabel("Power (W), 1 km average")
    ax[1].set_ylim(0, None)
    ax[1].legend(loc="lower left", frameon=False, ncol=2)
    ax[2].plot(km, real.t - opt.t, color="#2a78d6", lw=1.8)
    ax[2].axhline(0, color="#8a9098", lw=0.8)
    ax[2].set_ylabel("Optimal plan ahead by (s)")
    ax[2].set_xlabel("Distance (km)")
    ax[2].set_xlim(0, km[-1])
    fig.suptitle(f"70.3 Versailles bike leg: {clock(real.time)} ridden, {clock(opt.time)} with optimal pacing",
                 x=0.07, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def strategies_figure(res, path, title="70.3 Versailles bike leg: three ways to spend the same effort",
                      assume="Assumes 80 kg with the bike, drag area 0.30 m², no wind.", rode="What you rode (rebuilt from GPS)"):
    """One picture: what was ridden, steady power, and the optimum, at equal normalised power."""
    import matplotlib.pyplot as plt

    from ttopt import viz

    viz.style()
    mesh, c = res["mesh"], res["real"].cyclist
    steady = simulate(c, mesh, res["effort"], float(res["v"][0]), method="steady")
    plans = [
        (rode, res["real"], "#eb6834"),
        (f"Steady {res['effort']:.0f} W all the way", steady, "#1baf7a"),
        ("Optimal plan", res["same_np"], "#2a78d6"),
    ]
    km = mesh.s / 1000.0
    mid = 0.5 * (km[:-1] + km[1:])
    smooth = lambda y: _gaussian_smooth(np.asarray(y, float), DS, 300.0)
    ink, soft = "#1b1e22", "#555b63"

    fig = plt.figure(figsize=(12, 11.5))
    gs = fig.add_gridspec(5, 1, height_ratios=[1.25, 0.9, 2.1, 1.5, 1.3], hspace=0.32,
                          left=0.08, right=0.97, top=0.93, bottom=0.06)
    fig.suptitle(title, x=0.08, ha="left",
                 fontsize=15, fontweight="bold", color=ink)
    fig.text(0.08, 0.945, f"Same rider, same course, same normalised power ({res['effort']:.0f} W). "
             + assume, fontsize=10.5, color=soft, va="top")

    # finish times
    ax = fig.add_subplot(gs[0])
    base = res["real"].time
    lo = min(p.time for _, p, _ in plans) - 150.0
    for row, (name, plan, colour) in enumerate(plans):
        ax.barh(row, plan.time - lo, left=lo, height=0.56, color=colour)
        gap = plan.time - base
        note = "" if abs(gap) < 0.5 else f"   {abs(gap) / 60:.1f} min {'faster' if gap < 0 else 'slower'}"
        ax.text(plan.time + 6, row, clock(plan.time) + note, va="center", fontsize=11, fontweight="bold", color=ink)
        ax.text(lo - 6, row, name, va="center", ha="right", fontsize=10.5, color=ink)
    ax.set_ylim(len(plans) - 0.5, -0.5)
    ax.set_xlim(lo, max(p.time for _, p, _ in plans) + 0.9 * (max(p.time for _, p, _ in plans) - lo))
    ax.axis("off")
    ax.set_position([0.36, ax.get_position().y0, 0.61, ax.get_position().height])

    ax0 = fig.add_subplot(gs[1])
    ax0.fill_between(km, mesh.h, mesh.h.min() - 5, color="#8a9098", alpha=0.25, lw=0)
    ax0.plot(km, mesh.h, color=soft, lw=1.2)
    ax0.set_ylabel("Elevation (m)")

    ax1 = fig.add_subplot(gs[2], sharex=ax0)
    for name, plan, colour in plans:
        ax1.plot(mid, smooth(plan.power), color=colour, lw=1.9, label=name)
    ax1.set_ylabel("Power (W), 1 km average")
    ax1.set_ylim(0, None)
    ax1.legend(loc="lower left", frameon=False, ncol=3, fontsize=10)

    ax2 = fig.add_subplot(gs[3], sharex=ax0)
    for name, plan, colour in plans:
        ax2.plot(km, _gaussian_smooth(plan.v * 3.6, DS, 300.0), color=colour, lw=1.6)
    ax2.set_ylabel("Speed (km/h)")

    ax3 = fig.add_subplot(gs[4], sharex=ax0)
    for name, plan, colour in plans[1:]:
        gap = res["real"].t - plan.t
        ax3.plot(km, gap, color=colour, lw=1.9)
        ax3.text(km[-1] - 0.6, gap[-1] + (14 if gap[-1] > 0 else -14), f"{gap[-1]:+.0f} s", ha="right",
                 va="bottom" if gap[-1] > 0 else "top", fontsize=10, fontweight="bold", color=ink)
    ax3.axhline(0, color="#eb6834", lw=1.9)
    ax3.set_ylabel("Seconds ahead of you")
    ax3.set_xlabel("Distance (km)")
    ax3.set_xlim(0, km[-1])
    for a in (ax0, ax1, ax2):
        plt.setp(a.get_xticklabels(), visible=False)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return steady


def main():
    if "--strategies" in sys.argv:
        r = analyse(BASE, *load())
        steady = strategies_figure(r, ROOT / "figures" / "17_versailles_strategies.png")
        for name, plan in (("rode", r["real"]), ("steady", steady), ("optimal", r["same_np"])):
            print(f"{name:8s} {clock(plan.time)}  mean {plan.mean_power:.0f} W  p4 {fourth_power_mean(plan):.0f} W  max {plan.power.max():.0f} W")
        return
    course, position, t = load()
    print(f"course {course.length / 1000:.2f} km, ascent {course.total_ascent:.0f} m, "
          f"recorded {clock(t[-1])}, {3.6 * course.length / t[-1]:.1f} km/h")

    print("\nestimated power for different assumptions (W):")
    print("  mass  CdA   mean   NP  best20  best60  CP(refill fit)  braking%")
    for mass in (75.0, 80.0, 85.0):
        for cda in (0.26, 0.30, 0.34):
            r = analyse(BASE.with_(mass=mass, cda=cda), course, position, t, optimise=False)
            print(f"  {mass:4.0f} {cda:.2f}  {r['mean']:5.0f} {r['np']:4.0f}  {r['best20']:5.0f}   {r['best60']:5.0f}"
                  f"      {r['cp_recovery']:5.0f}        {100 * r['braking_share']:4.1f}")

    print("\noptimal pacing at equal effort:")
    print("  mass  CdA   rebuilt   same NP  gain   same work  gain   steady mean power")
    keep = None
    for mass, cda in ((80.0, 0.30), (80.0, 0.26), (80.0, 0.34), (75.0, 0.30), (85.0, 0.30)):
        r = analyse(BASE.with_(mass=mass, cda=cda), course, position, t)
        if keep is None:
            keep = r
        print(f"  {mass:4.0f} {cda:.2f}  {clock(r['real'].time)}  {clock(r['same_np'].time)} {r['real'].time - r['same_np'].time:5.0f}s"
              f"   {clock(r['same_work'].time)} {r['real'].time - r['same_work'].time:5.0f}s   {clock(r['steady'].time)}"
              f"   [replay error {r['real'].time - r['recorded']:+.0f}s, NP cap {r['effort']:.0f} W,"
              f" {r['same_np'].info['status']}]")

    print("\nhow the answer depends on the scale below which the GPS is not trusted (80 kg, 0.30 m2):")
    print("  scale(m)  ascent  mean   NP   rebuilt   same NP  gain")
    for scale in (150.0, 300.0, 600.0, 1000.0, 2000.0):
        r = analyse(BASE, course, position, t, scale=scale)
        print(f"  {scale:6.0f}   {r['ascent']:5.0f}  {r['mean']:4.0f}  {r['np']:4.0f}   {clock(r['real'].time)}  {clock(r['same_np'].time)}"
              f" {r['real'].time - r['same_np'].time:5.0f}s")

    # the steady wind that makes the ride look as even as possible: a lower bound on the gain
    best = min(((w, a) for w in np.arange(0.0, 8.1, 1.0) for a in np.arange(0.0, 360.0, 30.0)),
               key=lambda wa: (lambda q: q["p4"] / q["mean"])(analyse(BASE, course, position, t, optimise=False, wind=wa)))
    r = analyse(BASE, course, position, t, wind=best)
    print(f"\nmost favourable steady wind: {best[0] * 3.6:.0f} km/h from {best[1]:.0f} deg -> mean {r['mean']:.0f} W, NP {r['np']:.0f} W,"
          f" rebuilt {clock(r['real'].time)}, same NP {clock(r['same_np'].time)}, gain {r['real'].time - r['same_np'].time:.0f}s")

    mesh = at_scale(course, SCALE).mesh(DS, start_step=None)
    for watts in (150.0, 200.0, 250.0):
        print(f"steady {watts:.0f} W on this course (80 kg, 0.30 m2): {clock(simulate(BASE, mesh, watts, 5.0).time)}")

    r = keep
    real, opt = r["real"], r["same_np"]
    grade = np.diff(r["mesh"].h) / r["mesh"].ds * 100
    print("\nbase case, power by gradient (W):  gradient   you   optimal   share of time")
    for lo, hi in ((-99, -3), (-3, -1), (-1, 1), (1, 3), (3, 5), (5, 99)):
        pick = (grade >= lo) & (grade < hi)
        if real.dt[pick].sum() < 30:
            continue
        mine = (real.power[pick] * real.dt[pick]).sum() / real.dt[pick].sum()
        best = (opt.power[pick] * opt.dt[pick]).sum() / opt.dt[pick].sum()
        print(f"   {lo:>4} to {hi:<3}  {mine:5.0f}  {best:5.0f}    {100 * real.dt[pick].sum() / real.time:4.1f} %")
    figure(r, ROOT / "figures" / "16_versailles_vs_optimal.png")
    print("\nfigure: figures/16_versailles_vs_optimal.png")


if __name__ == "__main__":
    main()
