"""Shared physical model: the rider, the course, and one discrete energy balance.

Both solvers (direct collocation and dynamic programming) and the forward
simulator use the *same* discrete model, so their results are directly
comparable on a given mesh and all converge to the same continuous problem
when the mesh is refined.

Discrete model on one interval of road length ``ds`` (speed ``v0`` -> ``v1``):

    0.5 * m_i * (v1^2 - v0^2) = (F - B) * ds - E_drag - E_roll - E_grav

    E_drag = 0.5 * rho * CdA * ds * mean((v0 + w)^2, (v1 + w)^2)
    E_roll = Crr * m * g * cos(theta) * ds
    E_grav = m * g * dh

``F`` is the propulsive force at the wheel, ``B`` the braking force, ``w`` the
headwind. With the mean speed ``vb = (v0 + v1) / 2``:

    interval time      dt = ds / vb
    rider power        P  = F * vb / efficiency
    reserve burned     max(P - CP, 0) * dt      (never refilled)

The scheme is the work-energy theorem applied per interval, so gravity is
integrated exactly and energy is conserved by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
from numba import njit

G = 9.80665
V_FLOOR = 0.3  # m/s, lowest speed the model will represent


@dataclass(frozen=True)
class Cyclist:
    """A rider and their bike.

    cp        critical power (W): sustainable "for ever" in this model
    w_prime   anaerobic reserve (J): work available above cp, never refilled
    p_max     cap on instantaneous power (W)
    """

    name: str = "generic pro"
    mass: float = 78.0  # rider + bike (kg)
    cda: float = 0.22  # drag area (m^2)
    crr: float = 0.003  # rolling resistance coefficient
    cp: float = 410.0
    w_prime: float = 20_000.0
    p_max: float = 900.0
    efficiency: float = 0.975  # drivetrain
    f_max: float = 600.0  # cap on wheel force (N), matters only at the start
    brake_decel: float = 4.0  # m/s^2
    rotational_mass: float = 0.0  # extra inertia of the wheels (kg)

    @property
    def inertial_mass(self) -> float:
        return self.mass + self.rotational_mass

    @property
    def brake_max(self) -> float:
        return self.mass * self.brake_decel

    def with_(self, **changes) -> "Cyclist":
        return replace(self, **changes)


@dataclass(frozen=True)
class Mesh:
    """A course cut into N intervals. Node arrays have N+1 entries."""

    s: np.ndarray  # road distance at nodes (m)
    h: np.ndarray  # elevation at nodes (m)
    vmax: np.ndarray  # speed limit at nodes (m/s), inf where none
    wind: np.ndarray  # headwind on each interval (m/s)
    rho: float

    @property
    def n(self) -> int:
        return len(self.s) - 1

    @property
    def ds(self) -> np.ndarray:
        return np.diff(self.s)

    @property
    def dh(self) -> np.ndarray:
        return np.diff(self.h)

    @property
    def grade(self) -> np.ndarray:
        """sin(theta) on each interval."""
        return self.dh / self.ds

    @property
    def cos(self) -> np.ndarray:
        return np.sqrt(np.clip(1.0 - self.grade**2, 0.0, 1.0))


@dataclass
class Course:
    """An elevation profile along the road, plus optional extras.

    distance   road distance from the start (m), strictly increasing
    elevation  altitude (m) at those points
    v_limit    optional speed limit (m/s) at those points (e.g. corners)
    headwind   m/s, scalar or array on the same points (positive = against)
    """

    name: str
    distance: np.ndarray
    elevation: np.ndarray
    v_limit: Optional[np.ndarray] = None
    headwind: float | np.ndarray = 0.0
    rho: float = 1.2
    lat: Optional[np.ndarray] = None
    lon: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.distance = np.asarray(self.distance, dtype=float)
        self.elevation = np.asarray(self.elevation, dtype=float)
        if self.distance[0] != 0.0:
            self.distance = self.distance - self.distance[0]
        if np.any(np.diff(self.distance) <= 0):
            raise ValueError("distance must be strictly increasing")
        if self.v_limit is not None:
            self.v_limit = np.asarray(self.v_limit, dtype=float)

    @property
    def length(self) -> float:
        return float(self.distance[-1])

    @property
    def total_ascent(self) -> float:
        d = np.diff(self.elevation)
        return float(d[d > 0].sum())

    def mesh(self, ds: float = 50.0, start_step: float = 1.0, growth: float = 1.5, breaks=()) -> Mesh:
        """Mesh with intervals close to ``ds``, graded finely at the start.

        From a standing start the speed changes a lot over the first metres,
        so the first interval is ``start_step`` long and each following one is
        ``growth`` times longer until ``ds`` is reached. ``start_step=None``
        gives a plain uniform mesh. ``breaks`` are distances that must fall
        exactly on a node (timing checkpoints, for example).
        """
        head = [0.0]
        if start_step is not None and start_step < ds:
            step = start_step
            while step < ds and head[-1] + step < 0.5 * self.length:
                head.append(head[-1] + step)
                step *= growth
        stops = sorted(b for b in breaks if head[-1] + 1.0 < b < self.length - 1.0) + [self.length]
        pieces, at = [np.array(head[:-1])], head[-1]
        for stop in stops:
            n_piece = max(1, int(np.ceil((stop - at) / ds - 1e-9)))
            pieces.append(np.linspace(at, stop, n_piece + 1)[:-1])
            at = stop
        s = np.concatenate(pieces + [[self.length]])
        n = len(s) - 1
        h = np.interp(s, self.distance, self.elevation)
        if self.v_limit is None:
            vmax = np.full(n + 1, np.inf)
        else:
            # the limit at a node is the tightest limit within half a cell
            gaps = np.diff(s)
            before = 0.5 * np.concatenate([[0.0], gaps])
            after = 0.5 * np.concatenate([gaps, [0.0]])
            vmax = np.empty(n + 1)
            lo = np.searchsorted(self.distance, s - before, side="left")
            hi = np.searchsorted(self.distance, s + after, side="right")
            interp = np.interp(s, self.distance, self.v_limit)
            for i in range(n + 1):
                seg = self.v_limit[lo[i] : hi[i]]
                vmax[i] = min(interp[i], seg.min()) if seg.size else interp[i]
        mid = 0.5 * (s[:-1] + s[1:])
        if np.isscalar(self.headwind):
            wind = np.full(n, float(self.headwind))
        else:
            wind = np.interp(mid, self.distance, np.asarray(self.headwind, float))
        return Mesh(s=s, h=h, vmax=vmax, wind=wind, rho=self.rho)


# --------------------------------------------------------------------------
# The discrete energy balance (numpy version; the solvers mirror it)
# --------------------------------------------------------------------------


def required_wheel_force(c: Cyclist, mesh: Mesh, v: np.ndarray) -> np.ndarray:
    """Net wheel force (F - B) on each interval implied by node speeds ``v``."""
    v0, v1 = v[:-1], v[1:]
    ds = mesh.ds
    kd = 0.5 * mesh.rho * c.cda
    a0 = (v0 + mesh.wind) * np.abs(v0 + mesh.wind)
    a1 = (v1 + mesh.wind) * np.abs(v1 + mesh.wind)
    work = (
        0.5 * c.inertial_mass * (v1**2 - v0**2)
        + kd * ds * 0.5 * (a0 + a1)
        + c.crr * c.mass * G * mesh.cos * ds
        + c.mass * G * mesh.dh
    )
    return work / ds


@dataclass
class Strategy:
    """A pacing plan and the ride it produces."""

    method: str
    cyclist: Cyclist
    mesh: Mesh
    v: np.ndarray  # speed at nodes (m/s)
    power: np.ndarray  # rider power on each interval (W)
    brake: np.ndarray  # braking force on each interval (N)
    solve_seconds: float = 0.0
    info: dict = field(default_factory=dict)
    reserve: Optional[np.ndarray] = None  # reserve at nodes when it is allowed to refill

    @property
    def s(self) -> np.ndarray:
        return self.mesh.s

    @property
    def v_mid(self) -> np.ndarray:
        return 0.5 * (self.v[:-1] + self.v[1:])

    @property
    def dt(self) -> np.ndarray:
        return self.mesh.ds / self.v_mid

    @property
    def t(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(self.dt)])

    @property
    def time(self) -> float:
        return float(self.dt.sum())

    @property
    def w_burned(self) -> np.ndarray:
        """Reserve burned on each interval (J)."""
        return np.maximum(self.power - self.cyclist.cp, 0.0) * self.dt

    @property
    def w_bal(self) -> np.ndarray:
        """Reserve remaining at nodes (J)."""
        if self.reserve is not None:
            return self.reserve
        return self.cyclist.w_prime - np.concatenate([[0.0], np.cumsum(self.w_burned)])

    @property
    def mean_power(self) -> float:
        return float((self.power * self.dt).sum() / self.time)

    @property
    def mean_speed_kmh(self) -> float:
        return 3.6 * self.mesh.s[-1] / self.time

    def time_at(self, s: float | np.ndarray) -> np.ndarray:
        """Elapsed time at road distance ``s`` (linear in between nodes)."""
        return np.interp(s, self.mesh.s, self.t)

    def check(self) -> dict:
        """Residuals of the model equations and constraints for this plan."""
        c, m = self.cyclist, self.mesh
        net = required_wheel_force(c, m, self.v)
        f = self.power * c.efficiency / self.v_mid
        return {
            "force_residual_max_N": float(np.abs(net - (f - self.brake)).max()),
            "power_max_W": float(self.power.max()),
            "power_min_W": float(self.power.min()),
            "reserve_min_J": float(self.w_bal.min()),
            "speed_limit_excess_max": float(np.max(self.v - m.vmax)),
            "wheel_force_max_N": float(f.max()),
        }

    def summary(self) -> str:
        mm, ss = divmod(self.time, 60.0)
        return (
            f"{self.method:<12s} {int(mm):d}:{ss:06.3f}  "
            f"{self.mean_speed_kmh:5.2f} km/h  mean {self.mean_power:5.1f} W  "
            f"reserve low point {self.w_bal.min():7.0f} J  "
            f"[{self.solve_seconds:.2f} s to solve]"
        )


# --------------------------------------------------------------------------
# Forward simulation of a given power plan
# --------------------------------------------------------------------------


@njit(cache=True)
def _balance(v1, v0, p, ds, dh, cosk, w, m, mi, kd, crr, eta, fmax):
    """Energy left over (J) if the interval ends at v1; increasing in v1."""
    vb = 0.5 * (v0 + v1)
    f = eta * p / vb
    if f > fmax:
        f = fmax
    a0 = (v0 + w) * abs(v0 + w)
    a1 = (v1 + w) * abs(v1 + w)
    need = (
        0.5 * mi * (v1 * v1 - v0 * v0)
        + kd * ds * 0.5 * (a0 + a1)
        + crr * m * G * cosk * ds
        + m * G * dh
    )
    return need - f * ds


@njit(cache=True)
def _simulate(p, v_start, ds, dh, cosk, wind, vmax, m, mi, kd, crr, eta, fmax, vfloor):
    n = len(ds)
    v = np.empty(n + 1)
    power = np.empty(n)
    brake = np.zeros(n)
    stalled = 0
    v[0] = v_start
    for k in range(n):
        v0 = v[k]
        lo, hi = vfloor, 120.0
        if _balance(lo, v0, p[k], ds[k], dh[k], cosk[k], wind[k], m, mi, kd, crr, eta, fmax) > 0.0:
            v1 = lo  # cannot reach the next node at this power: stall
            stalled += 1
        else:
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if _balance(mid, v0, p[k], ds[k], dh[k], cosk[k], wind[k], m, mi, kd, crr, eta, fmax) > 0.0:
                    hi = mid
                else:
                    lo = mid
            v1 = 0.5 * (lo + hi)
        vb = 0.5 * (v0 + v1)
        f = min(eta * p[k] / vb, fmax)
        if v1 > vmax[k + 1]:
            v1 = vmax[k + 1]
            vb = 0.5 * (v0 + v1)
            f = min(eta * p[k] / vb, fmax)
            left = _balance(v1, v0, p[k], ds[k], dh[k], cosk[k], wind[k], m, mi, kd, crr, eta, fmax)
            brake[k] = -left / ds[k]
        v[k + 1] = v1
        power[k] = f * vb / eta
    return v, power, brake, stalled


def simulate(
    cyclist: Cyclist,
    mesh: Mesh,
    power: float | np.ndarray,
    v0: float = 1.0,
    method: str = "simulation",
) -> Strategy:
    """Ride the mesh with a prescribed power on each interval.

    Brakes are applied only where a speed limit requires it. The reserve is
    *not* enforced here: check ``strategy.w_bal`` to see whether the plan is
    affordable.
    """
    p = np.broadcast_to(np.asarray(power, dtype=float), (mesh.n,)).copy()
    c = cyclist
    v, pw, brake, stalled = _simulate(
        p, float(v0), mesh.ds, mesh.dh, mesh.cos, mesh.wind, mesh.vmax,
        c.mass, c.inertial_mass, 0.5 * mesh.rho * c.cda, c.crr, c.efficiency,
        c.f_max, V_FLOOR,
    )
    return Strategy(method, c, mesh, v, pw, brake, info={"stalled_intervals": int(stalled)})


def constant_power(cyclist: Cyclist, mesh: Mesh, v0: float = 1.0) -> Strategy:
    """Baseline: the constant power that empties the reserve exactly at the line."""
    c = cyclist

    def leftover(p):
        return simulate(c, mesh, p, v0).w_bal[-1]

    lo, hi = c.cp, c.p_max
    if leftover(hi) >= 0.0:
        p = hi
    else:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if leftover(mid) >= 0.0:
                lo = mid
            else:
                hi = mid
        p = lo
    out = simulate(c, mesh, p, v0, method="constant")
    out.info["power"] = p
    return out
