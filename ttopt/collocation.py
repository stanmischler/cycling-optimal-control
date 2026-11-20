"""Direct collocation: turn the pacing problem into one nonlinear programme.

Unknowns are the speed at every node and three forces on every interval:

    fb   propulsive force paid for by power up to CP      (fb * vb <= eta * CP)
    fa   propulsive force paid for by the reserve          (sum fa * ds / eta <= W')
    b    braking force

The split fb + fa removes the kink in ``max(P - CP, 0)``: because fb is free
and fa is not, the optimiser always fills fb first, so the split is exact
whenever the reserve is worth something. Since the reserve never refills, the
whole reserve history collapses into a single budget constraint.

The programme is solved with IPOPT through CasADi (exact derivatives).
"""

from __future__ import annotations

import time

import casadi as ca
import numpy as np

from .model import G, V_FLOOR, Cyclist, Mesh, Strategy, simulate


def _programme(c: Cyclist, mesh: Mesh, v0: float, cp, has_reserve: bool, budget: bool = True):
    """Variables, constraints and bounds shared by the two problems below.

    ``cp`` is a number, or a symbol when critical power is itself unknown.
    """
    n = mesh.n
    ds, dh, cosk, wind = mesh.ds, mesh.dh, mesh.cos, mesh.wind
    kd = 0.5 * mesh.rho * c.cda
    eta = c.efficiency

    v = ca.SX.sym("v", n + 1)
    fb = ca.SX.sym("fb", n)
    fa = ca.SX.sym("fa", n)
    b = ca.SX.sym("b", n)
    va, vb_ = v[:-1], v[1:]
    vm = 0.5 * (va + vb_)
    a0 = (va + wind) * ca.fabs(va + wind)
    a1 = (vb_ + wind) * ca.fabs(vb_ + wind)
    need = (
        0.5 * c.inertial_mass * (vb_**2 - va**2) / ds
        + kd * 0.5 * (a0 + a1)
        + c.crr * c.mass * G * cosk
        + c.mass * G * dh / ds
    )
    g = [
        need - (fb + fa - b),  # energy balance, in newtons (= 0)
        (fb * vm - eta * cp) / 100.0,
        ((fb + fa) * vm - eta * c.p_max) / 100.0,
        fb + fa - c.f_max,
    ]
    lbg = [np.zeros(n)] + [np.full(n, -np.inf)] * 3
    ubg = [np.zeros(n)] * 4
    if has_reserve and budget:
        g.append(ca.sum1(fa * ds) / (eta * c.w_prime) - 1.0)
        lbg.append(np.array([-np.inf]))
        ubg.append(np.array([0.0]))

    v_hi = np.where(np.isfinite(mesh.vmax), mesh.vmax, 60.0)
    lbx = np.concatenate([np.full(n + 1, V_FLOOR), np.zeros(3 * n)])
    ubx = np.concatenate(
        [v_hi, np.full(n, c.f_max), np.full(n, c.f_max if has_reserve else 0.0), np.full(n, c.brake_max)]
    )
    lbx[0] = ubx[0] = v0
    return {"v": v, "fb": fb, "fa": fa, "b": b, "vm": vm, "g": g, "lbg": lbg, "ubg": ubg,
            "lbx": lbx, "ubx": ubx, "v_hi": v_hi}


def _start_point(c: Cyclist, mesh: Mesh, v0: float, guess: Strategy, v_hi, cp: float):
    eta = c.efficiency
    gv = np.clip(guess.v, V_FLOOR, v_hi)
    gv[0] = v0
    gf = np.minimum(guess.power * eta / guess.v_mid, c.f_max)
    gfb = np.minimum(gf, eta * cp / guess.v_mid)
    return np.concatenate([gv, gfb, gf - gfb, np.clip(guess.brake, 0.0, c.brake_max)])


def _options(tol, max_iter, verbose):
    return {
        "ipopt.tol": tol, "ipopt.max_iter": max_iter,
        "ipopt.print_level": 5 if verbose else 0, "ipopt.sb": "yes", "print_time": False,
    }


def _unpack(c: Cyclist, mesh: Mesh, xs, method, elapsed, info) -> Strategy:
    n = mesh.n
    vs = xs[: n + 1]
    fbs, fas, bs = xs[n + 1 : 2 * n + 1], xs[2 * n + 1 : 3 * n + 1], xs[3 * n + 1 : 4 * n + 1]
    power = (fbs + fas) * 0.5 * (vs[:-1] + vs[1:]) / c.efficiency
    return Strategy(method, c, mesh, vs, power, np.maximum(bs, 0.0), elapsed, info)


def solve_collocation(
    cyclist: Cyclist,
    mesh: Mesh,
    v0: float = 1.0,
    guess: Strategy | None = None,
    tol: float = 1e-8,
    max_iter: int = 3000,
    verbose: bool = False,
    recovery: bool = False,
    effort_cap: float | None = None,
) -> Strategy:
    """Fastest ride for this rider on this mesh.

    effort_cap  if given (watts), critical power and the reserve are ignored
              and the ride is limited instead by its fourth-power mean power,
              (time-average of P^4)^(1/4) <= effort_cap. This is the idea
              behind "normalised power": hard surges cost much more than
              their average suggests. Used to compare with a recording at
              equal physiological cost.

    recovery  False (the rule of this project): reserve burned is gone.
              True: the reserve refills below critical power at the rate
              (CP - P) * (W' - W) / W' (Skiba), so the plan may surge and
              recover. The reserve then becomes a state along the course.
    """
    c = cyclist
    n = mesh.n
    if effort_cap is not None:
        c = c.with_(cp=c.p_max)  # all power is "free"; the cap below does the limiting
        recovery = False
    has_reserve = c.w_prime > 0.0 and c.p_max > c.cp
    recovery = recovery and has_reserve
    pr = _programme(c, mesh, v0, c.cp, has_reserve, budget=not recovery)
    g, lbg, ubg = list(pr["g"]), list(pr["lbg"]), list(pr["ubg"])
    lbx, ubx = pr["lbx"], pr["ubx"]
    parts = [pr["v"], pr["fb"], pr["fa"], pr["b"]]
    if recovery:
        # w = reserve / W' at nodes; burn fa*ds/eta, refill (CP - Pb) * (1 - w) * dt
        w = ca.SX.sym("w", n + 1)
        dt = mesh.ds / pr["vm"]
        pb = pr["fb"] * pr["vm"] / c.efficiency
        refill = (c.cp - pb) * (1.0 - w[:-1]) * dt / c.w_prime
        g.append(w[1:] - w[:-1] + pr["fa"] * mesh.ds / (c.efficiency * c.w_prime) - refill)
        lbg.append(np.zeros(n))
        ubg.append(np.zeros(n))
        parts.append(w)
        lbx = np.concatenate([lbx, [1.0], np.zeros(n)])
        ubx = np.concatenate([ubx, np.ones(n + 1)])
    if effort_cap is not None:
        dt = mesh.ds / pr["vm"]
        ratio = pr["fb"] * pr["vm"] / (c.efficiency * effort_cap)
        g.append(ca.sum1((ratio**4 - 1.0) * dt) / 1000.0)
        lbg.append(np.array([-np.inf]))
        ubg.append(np.array([0.0]))
    x = ca.vertcat(*parts)
    objective = ca.sum1(mesh.ds / pr["vm"])
    solver = ca.nlpsol("pacing", "ipopt", {"x": x, "f": objective, "g": ca.vertcat(*g)}, _options(tol, max_iter, verbose))

    if guess is None:
        guess = simulate(c, mesh, 0.95 * effort_cap if effort_cap is not None else c.cp, v0)
    x0 = _start_point(c, mesh, v0, guess, pr["v_hi"], c.cp)
    if recovery:
        x0 = np.concatenate([x0, np.ones(n + 1)])

    tic = time.perf_counter()
    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=np.concatenate(lbg), ubg=np.concatenate(ubg))
    elapsed = time.perf_counter() - tic

    stats = solver.stats()
    info = {
        "status": stats["return_status"],
        "success": bool(stats["success"]),
        "iterations": int(stats["iter_count"]),
        "objective": float(sol["f"]),
        "variables": int(x.shape[0]),
    }
    xs = np.asarray(sol["x"]).ravel()
    if effort_cap is not None:
        plan = _unpack(cyclist, mesh, xs, "collocation (equal effort)", elapsed, info)
        plan.reserve = np.full(n + 1, np.nan)
        return plan
    if recovery:
        plan = _unpack(c, mesh, xs, "collocation (reserve refills)", elapsed, info)
        plan.reserve = xs[4 * n + 1 :] * c.w_prime
        return plan
    if has_reserve:
        # seconds gained per extra kilojoule of reserve
        info["reserve_price_s_per_kJ"] = float(sol["lam_g"][-1]) / c.w_prime * 1000.0
    return _unpack(c, mesh, xs, "collocation", elapsed, info)


def most_economical_ride(
    cyclist: Cyclist,
    mesh: Mesh,
    nodes,
    times,
    v0: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 3000,
    verbose: bool = False,
) -> tuple[float, Strategy]:
    """The weakest rider who can hit given checkpoint times, and how they ride.

    Finds the lowest critical power for which a ride exists that passes node
    ``nodes[j]`` at time ``times[j]`` with the reserve ``cyclist.w_prime``.
    The ride returned is paced perfectly *inside* each segment; only the split
    of effort between segments is dictated by the times.

    Returns (critical power, ride).
    """
    c = cyclist
    cp = ca.SX.sym("cp")
    pr = _programme(c, mesh, v0, cp, True)
    dt = mesh.ds / pr["vm"]
    g, lbg, ubg = list(pr["g"]), list(pr["lbg"]), list(pr["ubg"])
    for node, target in zip(nodes, times):
        g.append(ca.sum1(dt[: int(node)]) / target - 1.0)
        lbg.append(np.array([0.0]))
        ubg.append(np.array([0.0]))
    x = ca.vertcat(pr["v"], pr["fb"], pr["fa"], pr["b"], cp)
    solver = ca.nlpsol("fit", "ipopt", {"x": x, "f": cp / 100.0, "g": ca.vertcat(*g)}, _options(tol, max_iter, verbose))

    # start from constant power per segment that reproduces the times
    from .real import segment_powers  # local import: real.py imports this module

    powers, guess = segment_powers(c, mesh, mesh.s[np.asarray(nodes, int)], times, v0)
    cp0 = float(np.min(powers))
    x0 = np.concatenate([_start_point(c.with_(f_max=c.f_max), mesh, v0, guess, pr["v_hi"], cp0), [cp0]])
    lbx = np.concatenate([pr["lbx"], [0.0]])
    ubx = np.concatenate([pr["ubx"], [c.p_max]])

    tic = time.perf_counter()
    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=np.concatenate(lbg), ubg=np.concatenate(ubg))
    elapsed = time.perf_counter() - tic
    xs = np.asarray(sol["x"]).ravel()
    stats = solver.stats()
    cp_fit = float(xs[-1])
    info = {"status": stats["return_status"], "success": bool(stats["success"]), "iterations": int(stats["iter_count"])}
    ride = _unpack(c.with_(cp=cp_fit), mesh, xs[:-1], "real (rebuilt)", elapsed, info)
    return cp_fit, ride
