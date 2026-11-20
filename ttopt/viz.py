"""Figures. One measure per panel, shared distance axis, no dual axes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

from .model import Strategy

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
# one-hue sequential ramp (light -> dark blue) for magnitudes such as power
POWER_CMAP = LinearSegmentedColormap.from_list(
    "power", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "font.family": "DejaVu Sans",
        "lines.linewidth": 2.0, "figure.dpi": 110, "savefig.dpi": 160,
    })


def _steps(plan: Strategy, values):
    """Interval values as a step line over distance in km."""
    return np.repeat(plan.s / 1000.0, 2)[1:-1], np.repeat(values, 2)


def fmt_time(seconds: float) -> str:
    m, s = divmod(seconds, 60.0)
    return f"{int(m)}:{s:04.1f}"


def _profile(ax, plan: Strategy, norm, power=None):
    """Elevation profile, filled with the plan's power."""
    power = plan.power if power is None else power
    x, h = plan.s / 1000.0, plan.mesh.h
    base = h.min() - 0.08 * max(np.ptp(h), 20.0)
    verts = [
        [(x[k], base), (x[k], h[k]), (x[k + 1], h[k + 1]), (x[k + 1], base)]
        for k in range(plan.mesh.n)
    ]
    poly = PolyCollection(verts, array=power, cmap=POWER_CMAP, norm=norm, edgecolors="face", linewidths=0.2)
    ax.add_collection(poly)
    ax.plot(x, h, color=INK, lw=1.0)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(base, h.max() + 0.12 * max(np.ptp(h), 20.0))
    ax.set_ylabel("Elevation (m)")
    ax.grid(False)
    return poly


def plot_plan(
    plan: Strategy,
    others: list[Strategy] | None = None,
    title: str | None = None,
    labels=None,
    checkpoints=None,
    checkpoint_labels=None,
    smooth_m: float = 0.0,
):
    """Elevation coloured by power, then power, speed and reserve along the course.

    ``others`` are overlaid as extra lines (for example a baseline or the
    other solver); the first plan sets the colours of the profile.
    ``checkpoints`` (road metres) are drawn as vertical guides.
    ``smooth_m`` averages the power lines over that many metres, which makes
    long real courses readable; the profile colours use the same average.
    """
    style()
    plans = [plan] + list(others or [])
    labels = labels or [p.method for p in plans]
    c = plan.cyclist
    fig, axes = plt.subplots(
        4, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.5, 1.2, 1.0], "hspace": 0.22}
    )
    lo = min(c.cp * 0.6, np.percentile(plan.power, 2))
    hi = max(c.cp * 1.25, np.percentile(plan.power, 99.5))
    norm = Normalize(lo, hi)
    def shown(p):
        if smooth_m <= 0:
            return p.power
        mid = 0.5 * (p.s[:-1] + p.s[1:])
        energy = np.concatenate([[0.0], np.cumsum(p.power * p.dt)])
        clock = p.t
        lo_i = np.searchsorted(p.s, mid - smooth_m / 2).clip(0, p.mesh.n)
        hi_i = np.searchsorted(p.s, mid + smooth_m / 2).clip(1, p.mesh.n)
        hi_i = np.maximum(hi_i, lo_i + 1)
        return (energy[hi_i] - energy[lo_i]) / (clock[hi_i] - clock[lo_i])

    poly = _profile(axes[0], plan, norm, shown(plan))
    cb = fig.colorbar(poly, ax=axes, location="right", fraction=0.025, pad=0.02, shrink=0.28, anchor=(0.0, 1.0))
    cb.set_label("Power (W)")
    cb.outline.set_visible(False)

    for i, (p, lab) in enumerate(zip(plans, labels)):
        col = SERIES[i % len(SERIES)]
        lw = 2.0 if i == 0 else 1.5
        axes[1].plot(*_steps(p, shown(p)), color=col, lw=lw, label=f"{lab}  {fmt_time(p.time)}")
        axes[2].plot(p.s / 1000.0, 3.6 * p.v, color=col, lw=lw)
        axes[3].plot(p.s / 1000.0, p.w_bal / 1000.0, color=col, lw=lw)
    axes[1].axhline(c.cp, color=INK_2, lw=1.0, ls=(0, (4, 3)))
    axes[1].annotate(
        f"critical power {c.cp:.0f} W", (plan.s[-1] / 1000.0, c.cp), xytext=(-4, -4),
        textcoords="offset points", ha="right", va="top", fontsize=8.5, color=INK_2,
    )
    top = np.percentile(np.concatenate([shown(p) for p in plans]), 99.7)
    axes[1].set_ylim(min(0.0, lo) if lo < 50 else max(0.0, lo - 40), max(top * 1.08, c.cp * 1.3))
    axes[1].set_ylabel("Power (W)")
    axes[2].set_ylabel("Speed (km/h)")
    axes[3].set_ylabel("Reserve left (kJ)")
    axes[3].set_ylim(bottom=0.0)
    axes[3].set_xlabel("Distance (km)")
    if len(plans) > 1:
        axes[1].legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=len(plans), borderaxespad=0.2, handlelength=1.6)
    if checkpoints is not None:
        names = checkpoint_labels or [""] * len(checkpoints)
        for idx, (where, name) in enumerate(zip(checkpoints, names)):
            for ax in axes:
                ax.axvline(where / 1000.0, color=INK_2, lw=0.8, ls=(0, (2, 3)), zorder=0.5)
            # alternate the height so neighbouring names do not collide
            axes[0].annotate(name, (where / 1000.0, 1.0), xycoords=("data", "axes fraction"), xytext=(-3, -3 - 12 * (idx % 2)),
                             textcoords="offset points", ha="right", va="top", fontsize=8.5, color=INK_2)
    axes[0].set_title(title or f"Optimal pacing, {fmt_time(plan.time)} at {plan.mean_speed_kmh:.1f} km/h")
    return fig


def plot_map(plan: Strategy, course, title: str | None = None):
    """The route seen from above, coloured by power."""
    style()
    if course.lat is None:
        raise ValueError("this course has no GPS coordinates")
    lat = np.interp(plan.s, course.distance, course.lat)
    lon = np.interp(plan.s, course.distance, course.lon)
    x = (lon - lon.mean()) * np.cos(np.radians(lat.mean())) * 111.32
    y = (lat - lat.mean()) * 110.57
    pts = np.column_stack([x, y])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    c = plan.cyclist
    norm = Normalize(min(c.cp * 0.6, np.percentile(plan.power, 2)), max(c.cp * 1.25, np.percentile(plan.power, 99.5)))
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.add_collection(LineCollection(segs, colors=INK, linewidths=5.6, capstyle="round"))
    lc = LineCollection(segs, array=plan.power, cmap=POWER_CMAP, norm=norm, linewidths=4.5, capstyle="round")
    ax.add_collection(lc)
    for i, name in ((0, "Start"), (-1, "Finish")):
        ax.plot(x[i], y[i], "o", ms=9, color=SURFACE, mec=INK, mew=1.6, zorder=5)
        ax.annotate(name, (x[i], y[i]), xytext=(8, 8), textcoords="offset points", fontsize=9.5, weight="bold")
    ax.set_aspect("equal")
    ax.autoscale()
    ax.margins(0.08)
    ax.set_xlabel("km east")
    ax.set_ylabel("km north")
    cb = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.02, shrink=0.6)
    cb.set_label("Power (W)")
    cb.outline.set_visible(False)
    ax.set_title(title or f"{course.name}: where the power goes")
    return fig


def plot_refinement(rows, title="Accuracy against computing time"):
    """Error on the finish time against solve time, one series per method."""
    style()
    fig, ax = plt.subplots(figsize=(8, 5.2))
    names = {"collocation": "Direct collocation", "dp": "Dynamic programming"}
    for i, method in enumerate(("collocation", "dp")):
        sub = sorted((r for r in rows if r["method"] == method), key=lambda r: r["seconds"])
        if not sub:
            continue
        t = np.array([r["seconds"] for r in sub])
        e = np.array([max(abs(r["error_s"]), 1e-5) for r in sub])
        ax.plot(t, e, "o", ms=7, color=SERIES[i], mec=SURFACE, mew=1.5, label=names[method], ls="none")
        # lower envelope: best accuracy reachable within a given time
        env = np.minimum.accumulate(e)
        ax.step(t, env, where="post", color=SERIES[i], lw=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time to solve (s)")
    ax.set_ylabel("Error on the finish time (s)")
    ax.legend(loc="upper right")
    ax.set_title(title)
    return fig


def plot_gain_scan(rows, race_titles=None):
    """Estimated gain against the assumed drag area, one panel per rider.

    One line per corner-grip assumption. Riders of one race share a row.
    """
    style()
    races = list(dict.fromkeys(r["race"] for r in rows))
    riders = {race: list(dict.fromkeys(r["rider"] for r in rows if r["race"] == race)) for race in races}
    ncol = max(len(v) for v in riders.values())
    grips = sorted({r["grip"] for r in rows})
    fig, axes = plt.subplots(len(races), ncol, figsize=(3.1 * ncol, 3.0 * len(races) + 0.4), sharey=True, sharex=True, squeeze=False)
    top = 0.0
    for i, race in enumerate(races):
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(riders[race]):
                ax.set_visible(False)
                continue
            rider = riders[race][j]
            for g, grip in enumerate(grips):
                sub = sorted(
                    (r for r in rows if r["race"] == race and r["rider"] == rider and r["grip"] == grip and r.get("gain_s") is not None),
                    key=lambda r: r["cda"],
                )
                if not sub:
                    continue
                y = [r["gain_s"] for r in sub]
                top = max(top, max(y))
                ax.plot([r["cda"] for r in sub], y, "-o", ms=5, color=SERIES[g], mec=SURFACE, mew=1.0,
                        label=f"corner grip {grip:g} g")
            ax.set_title(rider, fontsize=10)
            if j == 0:
                label = (race_titles or {}).get(race, race)
                ax.set_ylabel(f"{label}\nEstimated gain (s)")
            if i == len(races) - 1:
                ax.set_xlabel("Assumed drag area CdA (m²)")
    for ax in axes.ravel():
        ax.set_ylim(0, min(top * 1.05, 200))
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("The gain depends on what we assume about the rider", x=0.01, ha="left", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_replay_check(real: Strategy, position_m, speed_ms, clock_error_s, title="Does the model reproduce the real ride?"):
    """Recorded speed against the model fed the recorded power, and the clock drift."""
    style()
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.2})
    smooth = np.convolve(speed_ms, np.ones(5) / 5, "same")
    ax[0].plot(np.asarray(position_m) / 1000.0, 3.6 * smooth, color=SERIES[1], lw=1.5, label="Recorded speed")
    ax[0].plot(real.s / 1000.0, 3.6 * real.v, color=SERIES[0], lw=1.5, label="Model, fed the recorded power")
    ax[0].set_ylabel("Speed (km/h)")
    ax[0].legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=2, borderaxespad=0.2)
    ax[1].plot(real.s / 1000.0, clock_error_s, color=SERIES[0])
    ax[1].axhline(0, color=INK_2, lw=0.8)
    ax[1].set_ylabel("Model clock minus\nreal clock (s)")
    ax[1].set_xlabel("Distance (km)")
    ax[0].set_title(title, pad=22)
    return fig
