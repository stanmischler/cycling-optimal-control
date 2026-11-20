"""Dynamic programming on a (speed, reserve) grid.

State at each node: speed ``v`` and remaining reserve ``W``, each on a grid.
The decision on an interval is the speed at the next node, a *continuous*
number: the energy balance then gives the force, the power and the reserve
burned in closed form, and the value of the state reached is read from the
grid by bilinear interpolation.

Backward sweep (Bellman):  V_k(v, W) = min over v' of  dt + V_{k+1}(v', W - burned)
Forward sweep: start from the real state and re-minimise at every node with
the stored value function, carrying speed and reserve as continuous numbers,
so the ride returned satisfies the model exactly.

The inner minimisation samples the feasible range of v' (split at the speeds
where the rider coasts, rides exactly at critical power, and hits maximum
power), then polishes the best sample with a golden-section search.

The value function is also a feedback law: it gives the best action from
*any* speed and reserve at any point of the course, not just along one ride.
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit, prange

from .model import G, Cyclist, Mesh, Strategy

BIG = 1e17
N_BRAKE, N_FREE, N_BURN = 5, 8, 12  # samples in each part of the decision range
N_GOLD = 14  # golden-section iterations
INVPHI = 0.6180339887498949


@njit(cache=True, inline="always")
def _force(v1, v0, a0, wind, dsk, const, mi, kd):
    """Net wheel force needed to go from v0 to v1 over the interval."""
    a1 = (v1 + wind) * abs(v1 + wind)
    return 0.5 * mi * (v1 * v1 - v0 * v0) / dsk + kd * 0.5 * (a0 + a1) + const


@njit(cache=True, inline="always")
def _value(Vn, v1, w, v_bot, dvg, nv, dwg, nw):
    """Bilinear interpolation of the value function; BIG if not reachable."""
    x = (v1 - v_bot) / dvg
    i = int(x)
    if i >= nv - 1:
        i = nv - 2
    if i < 0:
        i = 0
    fx = x - i
    if nw == 1:
        a, b = Vn[i, 0], Vn[i + 1, 0]
        if fx <= 1e-12:
            return a
        if fx >= 1.0 - 1e-12:
            return b
        if a >= BIG or b >= BIG:
            return BIG
        return a + fx * (b - a)
    y = w / dwg
    j = int(y)
    if j >= nw - 1:
        j = nw - 2
    fy = y - j
    v00, v01, v10, v11 = Vn[i, j], Vn[i, j + 1], Vn[i + 1, j], Vn[i + 1, j + 1]
    # corners with zero weight are not needed
    need0 = fx < 1.0 - 1e-12
    need1 = fx > 1e-12
    lowy = fy < 1.0 - 1e-12
    highy = fy > 1e-12
    if need0 and ((lowy and v00 >= BIG) or (highy and v01 >= BIG)):
        return BIG
    if need1 and ((lowy and v10 >= BIG) or (highy and v11 >= BIG)):
        return BIG
    lo = v00 + fy * (v01 - v00) if highy else v00
    hi = v10 + fy * (v11 - v10) if highy else v10
    if not need1:
        return lo
    if not need0:
        return hi
    return lo + fx * (hi - lo)


@njit(cache=True, inline="always")
def _cubic(p0, p1, p2, p3, t):
    """Catmull-Rom interpolation between p1 and p2."""
    return p1 + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0)))


@njit(cache=True)
def _value_smooth(Vn, v1, w, v_bot, dvg, nv, dwg, nw):
    """Bicubic interpolation where a full 4x4 stencil exists, else bilinear.

    Bilinear interpolation makes the value function piecewise linear, and a
    piecewise-linear value makes the minimiser jump between grid lines (the
    power chatters). The backward sweep only needs values, so bilinear is
    fine there; the forward sweep needs a smooth minimiser.
    """
    x = (v1 - v_bot) / dvg
    i = int(x)
    y = w / dwg
    j = int(y)
    if nw < 4 or nv < 4 or i < 1 or i > nv - 3 or j < 1 or j > nw - 3:
        return _value(Vn, v1, w, v_bot, dvg, nv, dwg, nw)
    fx, fy = x - i, y - j
    col = np.empty(4)
    for a in range(4):
        r0, r1, r2, r3 = Vn[i - 1 + a, j - 1], Vn[i - 1 + a, j], Vn[i - 1 + a, j + 1], Vn[i - 1 + a, j + 2]
        if r0 >= BIG or r1 >= BIG or r2 >= BIG or r3 >= BIG:
            return _value(Vn, v1, w, v_bot, dvg, nv, dwg, nw)
        col[a] = _cubic(r0, r1, r2, r3, fy)
    return _cubic(col[0], col[1], col[2], col[3], fx)


@njit(cache=True, inline="always")
def _cost(v1, v0, w, a0, wind, dsk, const, mi, kd, eta, cp, Vn, v_bot, dvg, nv, dwg, nw, smooth):
    f = _force(v1, v0, a0, wind, dsk, const, mi, kd)
    vb = 0.5 * (v0 + v1)
    dt = dsk / vb
    burn = 0.0
    if f > 0.0:
        p = f * vb / eta
        if p > cp:
            burn = (p - cp) * dt
    wn = w - burn
    if wn < 0.0:
        if wn > -1e-9:
            wn = 0.0
        else:
            return BIG
    if smooth:
        nxt = _value_smooth(Vn, v1, wn, v_bot, dvg, nv, dwg, nw)
    else:
        nxt = _value(Vn, v1, wn, v_bot, dvg, nv, dwg, nw)
    if nxt >= BIG:
        return BIG
    return dt + nxt


@njit(cache=True)
def _anchors(v0, a0, wind, dsk, const, mi, kd, eta, cp, pmax, fmax, bmax, v_bot, v_top):
    """Decision range [lo, hi] and the speeds where power is 0 and CP.

    Force and power both increase with v1, so each bound is one bisection.
    Returns (ok, lo, coast, at_cp, hi).
    """
    # hi: largest v1 with force <= fmax and power <= pmax
    f = _force(v_top, v0, a0, wind, dsk, const, mi, kd)
    if f <= fmax and max(f, 0.0) * 0.5 * (v0 + v_top) / eta <= pmax:
        hi = v_top
    else:
        f = _force(v_bot, v0, a0, wind, dsk, const, mi, kd)
        if f > fmax or max(f, 0.0) * 0.5 * (v0 + v_bot) / eta > pmax:
            return False, 0.0, 0.0, 0.0, 0.0
        a, b = v_bot, v_top
        for _ in range(40):
            mid = 0.5 * (a + b)
            f = _force(mid, v0, a0, wind, dsk, const, mi, kd)
            if f <= fmax and max(f, 0.0) * 0.5 * (v0 + mid) / eta <= pmax:
                a = mid
            else:
                b = mid
        hi = a
    # lo: smallest v1 with braking force <= bmax
    if _force(v_bot, v0, a0, wind, dsk, const, mi, kd) >= -bmax:
        lo = v_bot
    else:
        if _force(hi, v0, a0, wind, dsk, const, mi, kd) < -bmax:
            return False, 0.0, 0.0, 0.0, 0.0
        a, b = v_bot, hi
        for _ in range(40):
            mid = 0.5 * (a + b)
            if _force(mid, v0, a0, wind, dsk, const, mi, kd) >= -bmax:
                b = mid
            else:
                a = mid
        lo = b
    # coast: largest v1 in [lo, hi] with force <= 0
    if _force(hi, v0, a0, wind, dsk, const, mi, kd) <= 0.0:
        coast = hi
    elif _force(lo, v0, a0, wind, dsk, const, mi, kd) > 0.0:
        coast = lo
    else:
        a, b = lo, hi
        for _ in range(40):
            mid = 0.5 * (a + b)
            if _force(mid, v0, a0, wind, dsk, const, mi, kd) <= 0.0:
                a = mid
            else:
                b = mid
        coast = a
    # at_cp: largest v1 in [coast, hi] with power <= cp
    f = _force(hi, v0, a0, wind, dsk, const, mi, kd)
    if max(f, 0.0) * 0.5 * (v0 + hi) / eta <= cp:
        at_cp = hi
    else:
        f = _force(coast, v0, a0, wind, dsk, const, mi, kd)
        if max(f, 0.0) * 0.5 * (v0 + coast) / eta > cp:
            at_cp = coast
        else:
            a, b = coast, hi
            for _ in range(40):
                mid = 0.5 * (a + b)
                f = _force(mid, v0, a0, wind, dsk, const, mi, kd)
                if max(f, 0.0) * 0.5 * (v0 + mid) / eta <= cp:
                    a = mid
                else:
                    b = mid
            at_cp = a
    return True, lo, coast, at_cp, hi


@njit(cache=True)
def _minimise(v0, w, lo, coast, at_cp, hi, a0, wind, dsk, const, mi, kd, eta, cp, Vn, v_bot, dvg, nv, dwg, nw, smooth):
    """Best next speed from (v0, w). Returns (cost, v1)."""
    nc = N_BRAKE + N_FREE + N_BURN + 1
    cand = np.empty(nc)
    m = 0
    for q in range(N_BRAKE):
        cand[m] = lo + (coast - lo) * q / N_BRAKE
        m += 1
    for q in range(N_FREE):
        cand[m] = coast + (at_cp - coast) * q / N_FREE
        m += 1
    for q in range(N_BURN + 1):
        cand[m] = at_cp + (hi - at_cp) * q / N_BURN
        m += 1
    best, bi = BIG, -1
    vals = np.empty(nc)
    for q in range(nc):
        vals[q] = _cost(cand[q], v0, w, a0, wind, dsk, const, mi, kd, eta, cp, Vn, v_bot, dvg, nv, dwg, nw, smooth)
        if vals[q] < best:
            best, bi = vals[q], q
    if bi < 0:
        return BIG, v0
    bv = cand[bi]
    # polish on each side of the best sample (the kink at critical power is a sample)
    for side in range(2):
        if side == 0:
            if bi == 0:
                continue
            a, b = cand[bi - 1], cand[bi]
        else:
            if bi == nc - 1:
                continue
            a, b = cand[bi], cand[bi + 1]
        if b - a < 1e-9:
            continue
        c = b - INVPHI * (b - a)
        d = a + INVPHI * (b - a)
        fc = _cost(c, v0, w, a0, wind, dsk, const, mi, kd, eta, cp, Vn, v_bot, dvg, nv, dwg, nw, smooth)
        fd = _cost(d, v0, w, a0, wind, dsk, const, mi, kd, eta, cp, Vn, v_bot, dvg, nv, dwg, nw, smooth)
        for _ in range(N_GOLD):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - INVPHI * (b - a)
                fc = _cost(c, v0, w, a0, wind, dsk, const, mi, kd, eta, cp, Vn, v_bot, dvg, nv, dwg, nw, smooth)
            else:
                a, c, fc = c, d, fd
                d = a + INVPHI * (b - a)
                fd = _cost(d, v0, w, a0, wind, dsk, const, mi, kd, eta, cp, Vn, v_bot, dvg, nv, dwg, nw, smooth)
        if fc < best:
            best, bv = fc, c
        if fd < best:
            best, bv = fd, d
    return best, bv


@njit(cache=True, parallel=True)
def _backward(vg, wg, ds, dh, cosk, wind, vmax, m, mi, kd, crr, eta, cp, pmax, fmax, bmax):
    n, nv, nw = len(ds), len(vg), len(wg)
    dvg = vg[1] - vg[0]
    dwg = wg[1] - wg[0] if nw > 1 else 1.0
    v_bot = vg[0]
    V = np.empty((n + 1, nv, nw))
    for i in range(nv):
        for j in range(nw):
            V[n, i, j] = 0.0 if vg[i] <= vmax[n] + 1e-9 else BIG
    for k in range(n - 1, -1, -1):
        Vn = V[k + 1]
        dsk = ds[k]
        const = crr * m * G * cosk[k] + m * G * dh[k] / dsk
        v_top = min(vg[nv - 1], vmax[k + 1])
        for i in prange(nv):
            v0 = vg[i]
            if v0 > vmax[k] + 1e-9:
                for j in range(nw):
                    V[k, i, j] = BIG
                continue
            a0 = (v0 + wind[k]) * abs(v0 + wind[k])
            ok, lo, coast, at_cp, hi = _anchors(
                v0, a0, wind[k], dsk, const, mi, kd, eta, cp, pmax, fmax, bmax, v_bot, v_top
            )
            for j in range(nw):
                if not ok:
                    V[k, i, j] = BIG
                else:
                    V[k, i, j] = _minimise(
                        v0, wg[j], lo, coast, at_cp, hi, a0, wind[k], dsk, const,
                        mi, kd, eta, cp, Vn, v_bot, dvg, nv, dwg, nw, False,
                    )[0]
    return V


@njit(cache=True)
def _forward(V, v_start, w0, vg, wg, ds, dh, cosk, wind, vmax, m, mi, kd, crr, eta, cp, pmax, fmax, bmax):
    n, nv, nw = len(ds), len(vg), len(wg)
    dvg = vg[1] - vg[0]
    dwg = wg[1] - wg[0] if nw > 1 else 1.0
    v_bot = vg[0]
    v = np.empty(n + 1)
    power = np.zeros(n)
    brake = np.zeros(n)
    v[0] = v_start
    w = w0
    for k in range(n):
        v0 = v[k]
        dsk = ds[k]
        const = crr * m * G * cosk[k] + m * G * dh[k] / dsk
        v_top = min(vg[nv - 1], vmax[k + 1])
        a0 = (v0 + wind[k]) * abs(v0 + wind[k])
        ok, lo, coast, at_cp, hi = _anchors(
            v0, a0, wind[k], dsk, const, mi, kd, eta, cp, pmax, fmax, bmax, v_bot, v_top
        )
        if not ok:
            return v, power, brake, k
        cost, v1 = _minimise(
            v0, w, lo, coast, at_cp, hi, a0, wind[k], dsk, const, mi, kd, eta, cp,
            V[k + 1], v_bot, dvg, nv, dwg, nw, True,
        )
        if cost >= BIG:
            return v, power, brake, k
        f = _force(v1, v0, a0, wind[k], dsk, const, mi, kd)
        vb = 0.5 * (v0 + v1)
        if f >= 0.0:
            power[k] = f * vb / eta
            if power[k] > cp:
                w = max(w - (power[k] - cp) * dsk / vb, 0.0)
        else:
            brake[k] = -f
        v[k + 1] = v1
    return v, power, brake, n


def solve_dp(
    cyclist: Cyclist,
    mesh: Mesh,
    v0: float = 1.0,
    dv: float = 0.1,
    n_w: int = 81,
    v_top: float = 30.0,
    memory_limit_gb: float = 6.0,
) -> Strategy:
    """Dynamic programming solution.

    dv      spacing of the speed grid (m/s)
    n_w     number of points on the reserve grid
    v_top   highest speed on the grid when the course has no speed limit
    """
    c = cyclist
    finite = mesh.vmax[np.isfinite(mesh.vmax)]
    v_hi = min(v_top, finite.max()) if finite.size == mesh.vmax.size else v_top
    v_lo = min(v0, 1.0)
    nv = max(2, int(np.ceil((v_hi - v_lo) / dv - 1e-9)) + 1)
    vg = v_lo + dv * np.arange(nv)
    if c.w_prime <= 0.0 or n_w < 2:
        wg = np.array([0.0])
    else:
        wg = np.linspace(0.0, c.w_prime, n_w)

    gb = (mesh.n + 1) * nv * len(wg) * 8 / 1e9
    if gb > memory_limit_gb:
        raise MemoryError(
            f"value function would need {gb:.1f} GB "
            f"({mesh.n + 1} nodes x {nv} speeds x {len(wg)} reserve levels); "
            "use a coarser grid or raise memory_limit_gb"
        )

    args = (
        vg, wg, mesh.ds, mesh.dh, mesh.cos, mesh.wind, mesh.vmax,
        c.mass, c.inertial_mass, 0.5 * mesh.rho * c.cda, c.crr, c.efficiency,
        c.cp, c.p_max, c.f_max, c.brake_max,
    )
    tic = time.perf_counter()
    V = _backward(*args)
    v, power, brake, reached = _forward(V, float(v0), float(c.w_prime), *args)
    elapsed = time.perf_counter() - tic
    if reached < mesh.n:
        raise RuntimeError(f"dynamic programme found no feasible ride past node {reached}")

    i0 = int(round((v0 - v_lo) / dv))
    info = {
        "value_at_start": float(V[0, i0, -1]),
        "speed_grid": nv,
        "reserve_grid": len(wg),
        "states": int((mesh.n + 1) * nv * len(wg)),
        "memory_gb": gb,
        "value_function": V,
        "grids": (vg, wg),
    }
    return Strategy("dp", c, mesh, v, power, brake, elapsed, info)
