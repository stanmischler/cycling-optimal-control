"""Compare the optimal plan with what a real rider did.

We rarely have a rider's full power file, but official timing gives their
time at a few checkpoints. From those we rebuild the simplest ride that
reproduces them: one constant power per segment between checkpoints. The
rider's critical power is then set so that this ride spends exactly their
reserve, and the optimiser is given the same rider on the same course. The
difference in finish time is the estimated gain from better pacing.

The estimate is conditional on the assumed drag area, mass and reserve, so
``compare`` should be run over a range of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import solve
from .model import Course, Cyclist, Mesh, Strategy, simulate


def segment_powers(
    cyclist: Cyclist, mesh: Mesh, checkpoints_m, times_s, v0: float = 1.0, p_hi: float = 1500.0
) -> tuple[np.ndarray, Strategy]:
    """Constant power on each segment that reproduces the checkpoint times.

    checkpoints_m  road distance of each checkpoint, the last being the finish
    times_s        elapsed time at each checkpoint
    Returns (power per segment, simulated ride).
    """
    checkpoints_m = np.asarray(checkpoints_m, float)
    times_s = np.asarray(times_s, float)
    mid = 0.5 * (mesh.s[:-1] + mesh.s[1:])
    seg = np.searchsorted(checkpoints_m, mid, side="left").clip(max=len(checkpoints_m) - 1)
    wide = cyclist.with_(p_max=p_hi)
    powers = np.full(len(checkpoints_m), cyclist.cp)
    for j, (where, target) in enumerate(zip(checkpoints_m, times_s)):
        lo, hi = 0.0, p_hi
        for _ in range(50):
            powers[j] = 0.5 * (lo + hi)
            ride = simulate(wide, mesh, powers[seg], v0)
            if ride.time_at(where) > target:
                lo = powers[j]
            else:
                hi = powers[j]
        powers[j] = 0.5 * (lo + hi)
    ride = simulate(wide, mesh, powers[seg], v0, method="real (rebuilt)")
    ride.info["segment_powers"] = powers.copy()
    ride.info["checkpoint_error_s"] = ride.time_at(checkpoints_m) - times_s
    return powers, ride


def critical_power_for(ride: Strategy, w_prime: float) -> float:
    """Critical power at which ``ride`` burns exactly ``w_prime`` joules."""
    lo, hi = 0.0, float(ride.power.max())
    for _ in range(60):
        cp = 0.5 * (lo + hi)
        burned = (np.maximum(ride.power - cp, 0.0) * ride.dt).sum()
        lo, hi = (cp, hi) if burned > w_prime else (lo, cp)
    return 0.5 * (lo + hi)


@dataclass
class Comparison:
    cyclist: Cyclist  # with the fitted critical power
    real: Strategy
    optimal: Strategy
    checkpoints_m: np.ndarray
    times_s: np.ndarray

    @property
    def gain_s(self) -> float:
        return self.real.time - self.optimal.time

    def segment_table(self):
        """Per segment: real and optimal time and mean power."""
        rows, start = [], 0.0
        for stop in self.checkpoints_m:
            row = {"from_km": start / 1000.0, "to_km": stop / 1000.0}
            for label, plan in (("real", self.real), ("optimal", self.optimal)):
                mid = 0.5 * (plan.s[:-1] + plan.s[1:])
                m = (mid > start) & (mid <= stop)
                row[f"{label}_time_s"] = float(plan.dt[m].sum())
                row[f"{label}_power_W"] = float((plan.power[m] * plan.dt[m]).sum() / plan.dt[m].sum())
            rows.append(row)
            start = stop
        return rows


def compare(
    cyclist: Cyclist,
    course: Course,
    checkpoints_m,
    times_s,
    rebuild: str = "economical",
    ds: float = 50.0,
    v0: float = 1.0,
) -> Comparison:
    """Rebuild the real ride from checkpoint times, then optimise the same rider.

    rebuild  "economical": the real rider is assumed to pace perfectly inside
             each segment, so only the split of effort *between* segments is
             judged. This is the fairest reading of a handful of split times
             and gives a cautious (low) estimate of the gain.
             "constant": one constant power per segment. Simple, but it also
             charges the rider for not modulating power inside a segment,
             which they certainly did, so it overstates the gain.

    ``cyclist.cp`` is ignored and refitted so that the rebuilt ride spends
    exactly the reserve; mass, drag area, rolling resistance, reserve and
    maximum power are taken as given.
    """
    from .collocation import most_economical_ride

    checkpoints_m = np.asarray(checkpoints_m, float)
    times_s = np.asarray(times_s, float)
    mesh = course.mesh(ds, breaks=checkpoints_m[:-1])
    nodes = np.array([int(np.argmin(np.abs(mesh.s - c))) for c in checkpoints_m])
    if rebuild == "constant":
        powers, real = segment_powers(cyclist, mesh, checkpoints_m, times_s, v0)
        cp = critical_power_for(real, cyclist.w_prime)
        fitted = cyclist.with_(cp=cp, p_max=max(cyclist.p_max, 1.2 * powers.max()))
        real.cyclist = fitted
    elif rebuild == "economical":
        cp, real = most_economical_ride(cyclist, mesh, nodes, times_s, v0)
        if not real.info["success"]:
            raise RuntimeError(f"could not rebuild the ride: {real.info['status']}")
        fitted = real.cyclist
    else:
        raise ValueError(f"unknown rebuild {rebuild!r}")
    optimal = solve(fitted, mesh, "collocation", v0=v0)
    return Comparison(fitted, real, optimal, checkpoints_m, times_s)


# --------------------------------------------------------------------------
# With a real power file
# --------------------------------------------------------------------------


def power_on_mesh(mesh: Mesh, distance_m, power_w, time_s) -> tuple[np.ndarray, np.ndarray]:
    """Measured power averaged over each mesh interval, and real time at nodes.

    The average is taken over the time the rider actually spent in the
    interval, so the energy of the recording is preserved.
    """
    d = np.asarray(distance_m, float)
    t = np.asarray(time_s, float)
    p = np.asarray(power_w, float)
    d = np.maximum.accumulate(d)  # distance never goes backwards
    energy = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(t))])
    # time at which the rider passes each node (first passage)
    grow = np.concatenate([[True], np.diff(d) > 0])
    t_nodes = np.interp(mesh.s, d[grow], t[grow])
    e_nodes = np.interp(t_nodes, t, energy)
    dt = np.diff(t_nodes)
    return np.diff(e_nodes) / np.maximum(dt, 1e-9), t_nodes


@dataclass
class FileComparison:
    cyclist: Cyclist  # drag area and critical power fitted to the recording
    real: Strategy  # the measured power replayed through the model
    optimal: Strategy
    recorded_time_at_nodes: np.ndarray

    @property
    def gain_s(self) -> float:
        return self.real.time - self.optimal.time

    @property
    def replay_error_s(self) -> np.ndarray:
        """Model clock minus real clock along the course (after calibration)."""
        return self.real.t - (self.recorded_time_at_nodes - self.recorded_time_at_nodes[0])


def fourth_power_mean(plan: Strategy) -> float:
    """(time-average of P^4)^(1/4): a normalised-power style cost of a ride."""
    return float(((plan.power**4 * plan.dt).sum() / plan.time) ** 0.25)


def reserve_with_recovery(power, dt, cp: float, w_prime: float) -> np.ndarray:
    """Reserve left over time when it *does* refill below critical power.

    Differential form of the Skiba W'bal model: above CP the reserve drains
    at (P - CP); below CP it refills at (CP - P) * (W' - W) / W', so recovery
    slows down as the tank fills. Used only to read a real recording: a real
    power trace surges far above CP hundreds of times, which is impossible
    without recovery.
    """
    w = np.empty(len(power) + 1)
    w[0] = w_prime
    for k, (p, step) in enumerate(zip(power, dt)):
        if p > cp:
            w[k + 1] = w[k] - (p - cp) * step
        else:
            w[k + 1] = w_prime - (w_prime - w[k]) * np.exp(-(cp - p) * step / w_prime)
    return w


def critical_power_with_recovery(power, dt, w_prime: float) -> float:
    """Lowest critical power for which the recording never empties the reserve."""
    lo, hi = 0.0, float(np.max(power))
    for _ in range(50):
        cp = 0.5 * (lo + hi)
        if reserve_with_recovery(power, dt, cp, w_prime).min() < 0.0:
            lo = cp
        else:
            hi = cp
    return hi


def compare_power_file(
    cyclist: Cyclist,
    course: Course,
    distance_m,
    power_w,
    time_s,
    v0: float | None = None,
    ds: float = 50.0,
    fit: str | None = "cda",
    recovery: bool = True,
) -> FileComparison:
    """Replay a recorded ride, then optimise the same rider on the same course.

    1. The measured power is averaged onto the course mesh.
    2. One model parameter is calibrated so that replaying that power gives
       the real finish time: the drag area (``fit="cda"``) or the mass
       (``fit="mass"``); ``fit=None`` keeps ``cyclist`` as given.
    3. Critical power is read off the 1 Hz recording with a reserve that
       refills below critical power (``critical_power_with_recovery``):
       the lowest value the recording is compatible with.
    4. The optimiser plans the fastest ride for that rider. With
       ``recovery=True`` it plays by the same rule as the recording (the
       reserve refills), which is the like-for-like comparison. With
       ``recovery=False`` it uses the stricter no-refill rule of the rest of
       this project, and can then be *slower* than the real rider.

    ``v0`` is the speed at the first sample (taken from the recording if None).
    """
    d = np.asarray(distance_m, float)
    t = np.asarray(time_s, float)
    raw_power = np.asarray(power_w, float)
    if v0 is None:
        v0 = max((d[min(5, len(d) - 1)] - d[0]) / max(t[min(5, len(t) - 1)] - t[0], 1e-9), 0.5)
    mesh = course.mesh(ds)
    power, t_nodes = power_on_mesh(mesh, d - d[0], raw_power, t)
    target = t_nodes[-1] - t_nodes[0]
    fitted = cyclist.with_(p_max=max(cyclist.p_max, 1.05 * power.max()), f_max=max(cyclist.f_max, 2000.0))

    if fit is not None:
        lo, hi = (0.05, 0.8) if fit == "cda" else (40.0, 140.0)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            trial = fitted.with_(cda=mid) if fit == "cda" else fitted.with_(mass=mid)
            if simulate(trial, mesh, power, v0).time < target:
                lo = mid  # too fast: more drag or more mass
            else:
                hi = mid
        value = 0.5 * (lo + hi)
        fitted = fitted.with_(cda=value) if fit == "cda" else fitted.with_(mass=value)
    real = simulate(fitted, mesh, power, v0, method="real (power file)")
    cp = critical_power_with_recovery(raw_power, np.gradient(t), cyclist.w_prime)
    fitted = fitted.with_(cp=cp)
    real.cyclist = fitted
    real.info["cp_without_recovery"] = critical_power_for(real, cyclist.w_prime)
    recorded = reserve_with_recovery(raw_power, np.gradient(t), cp, cyclist.w_prime)
    real.reserve = np.interp(t_nodes, np.concatenate([t, [t[-1] + 1.0]]), recorded)
    optimal = solve(fitted, mesh, "collocation", v0=v0, recovery=recovery)
    return FileComparison(fitted, real, optimal, t_nodes)


def speed_limits_from_recording(course: Course, position_m, speed_ms, power_w, coast_below=30.0, margin=1.03):
    """Speed limits taken from what the rider actually did.

    Wherever the rider stopped pedalling (corners, fast descents), the limit
    is the speed they held there; elsewhere it is their top speed of the day.
    The optimiser may therefore never corner or descend faster than the real
    rider did, which keeps the comparison about pacing and not about bike
    handling.
    """
    pos = np.asarray(position_m, float)
    v = np.asarray(speed_ms, float)
    coasting = np.asarray(power_w, float) < coast_below
    # widen each coasting stretch by 3 samples: braking starts just before
    wide = coasting.copy()
    for shift in range(1, 4):
        wide[shift:] |= coasting[:-shift]
        wide[:-shift] |= coasting[shift:]
    top = float(v.max()) * margin
    limit = np.full(len(course.distance), top)
    if wide.any():
        idx = np.flatnonzero(wide)
        near = np.interp(course.distance, pos, np.arange(len(pos)))
        lo = np.searchsorted(idx, near - 2.0)
        hi = np.searchsorted(idx, near + 2.0)
        for i in np.flatnonzero(hi > lo):
            limit[i] = v[idx[lo[i] : hi[i]]].max() * margin
    return limit
