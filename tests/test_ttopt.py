import numpy as np
import pytest

from ttopt import Course, Cyclist, courses, simulate, solve
from ttopt.model import G

RIDER = Cyclist()


def steady_speed(c, power, grade=0.0, rho=1.2):
    """Root of eta*P = v * (0.5 rho CdA v^2 + m g (crr cos + sin))."""
    cos = np.sqrt(1 - grade**2)
    coeffs = [0.5 * rho * c.cda, 0.0, c.mass * G * (c.crr * cos + grade), -c.efficiency * power]
    roots = np.roots(coeffs)
    return float(max(r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 0))


def test_simulation_reaches_analytic_steady_speed():
    ride = simulate(RIDER, courses.flat(20_000).mesh(20), 350.0)
    assert ride.v[-1] == pytest.approx(steady_speed(RIDER, 350.0), rel=1e-6)


def test_simulation_steady_speed_on_a_climb():
    x = np.linspace(0, 20_000, 2001)
    course = Course("ramp", x, 0.06 * x / np.sqrt(1 + 0.06**2))  # road distance vs height
    ride = simulate(RIDER, course.mesh(20), 400.0)
    grade = course.mesh(20).grade[0]
    assert ride.v[-1] == pytest.approx(steady_speed(RIDER, 400.0, grade), rel=1e-6)


def test_energy_is_conserved_by_the_simulator():
    mesh = courses.rolling().mesh(50)
    ride = simulate(RIDER, mesh, 380.0)
    c = RIDER
    work_in = (ride.power * ride.dt).sum() * c.efficiency
    kinetic = 0.5 * c.mass * (ride.v[-1] ** 2 - ride.v[0] ** 2)
    potential = c.mass * G * (mesh.h[-1] - mesh.h[0])
    drag = (0.5 * mesh.rho * c.cda * mesh.ds * 0.5 * (ride.v[:-1] ** 2 + ride.v[1:] ** 2)).sum()
    rolling = (c.crr * c.mass * G * mesh.cos * mesh.ds).sum()
    assert work_in == pytest.approx(kinetic + potential + drag + rolling, rel=1e-9)


@pytest.mark.parametrize("make", list(courses.SYNTHETIC.values()))
def test_collocation_is_feasible_and_beats_constant_power(make):
    course = make()
    plan = solve(RIDER, course, "collocation")
    base = solve(RIDER, course, "constant")
    chk = plan.check()
    assert plan.info["success"]
    assert chk["force_residual_max_N"] < 1e-5
    assert chk["power_max_W"] <= RIDER.p_max + 1e-3
    assert chk["reserve_min_J"] > -1e-2
    assert plan.time < base.time


def test_without_reserve_the_rider_sits_at_critical_power():
    rider = RIDER.with_(w_prime=0.0)
    mesh = courses.single_hill().mesh(50)
    plan = solve(rider, mesh, "collocation")
    assert np.allclose(plan.power, rider.cp, atol=1e-3)
    assert plan.time == pytest.approx(simulate(rider, mesh, rider.cp).time, rel=1e-7)


def test_power_never_drops_below_critical_power_without_speed_limits():
    # power up to CP is free and always helps, so the optimum uses all of it
    plan = solve(RIDER, courses.single_hill(), "collocation")
    assert plan.power.min() > RIDER.cp - 0.05


def test_flat_course_is_ridden_at_constant_speed_in_the_middle():
    plan = solve(RIDER, courses.flat(20_000), "collocation")
    middle = plan.v[len(plan.v) // 4 : 3 * len(plan.v) // 4]
    assert np.ptp(middle) < 0.02


def test_more_power_uphill_than_on_the_flat():
    plan = solve(RIDER, courses.single_hill(), "collocation")
    s_mid = 0.5 * (plan.s[:-1] + plan.s[1:])
    climb = plan.power[(s_mid > 4200) & (s_mid < 5800)].mean()
    flat_before = plan.power[(s_mid > 1500) & (s_mid < 3500)].mean()
    assert climb > flat_before + 20


@pytest.mark.parametrize("make", list(courses.SYNTHETIC.values()))
def test_dp_agrees_with_collocation(make):
    mesh = make().mesh(50)
    col = solve(RIDER, mesh, "collocation")
    dp = solve(RIDER, mesh, "dp")
    assert dp.check()["reserve_min_J"] >= -1e-6
    assert dp.check()["force_residual_max_N"] < 1e-8
    # same discrete model, two independent solvers: they must land on the same time
    assert dp.time == pytest.approx(col.time, abs=0.2)
    # the rides coincide except on a few intervals where the reserve runs out
    assert np.abs(dp.v - col.v).mean() < 0.05
    assert np.median(np.abs(dp.power - col.power)) < 3.0


def test_dp_converges_towards_collocation_when_its_grid_is_refined():
    mesh = courses.single_hill().mesh(50)
    col = solve(RIDER, mesh, "collocation").time
    coarse = abs(solve(RIDER, mesh, "dp", dv=1.0, n_w=6).time - col)
    fine = abs(solve(RIDER, mesh, "dp", dv=0.1, n_w=41).time - col)
    assert fine < 0.5 * coarse
    assert fine < 0.05


def test_collocation_converges_under_mesh_refinement():
    course = courses.single_hill()
    t = [solve(RIDER, course, "collocation", ds=ds).time for ds in (200, 100, 50, 25)]
    steps = np.abs(np.diff(t))
    assert steps[-1] < steps[0]
    assert steps[-1] < 0.05


def test_speed_limit_is_respected_and_needs_the_brakes():
    course = courses.climb_and_descent(gradient_down=0.09)
    course.v_limit = np.where((course.distance > 8_000) & (course.distance < 8_300), 8.0, 25.0)
    mesh = course.mesh(50)
    for method in ("collocation", "dp"):
        plan = solve(RIDER, mesh, method)
        assert np.all(plan.v <= mesh.vmax + 1e-6)
        assert plan.brake.max() > 10.0


def test_a_stronger_rider_is_faster():
    course = courses.rolling()
    assert solve(RIDER.with_(cp=430.0), course).time < solve(RIDER, course).time
    assert solve(RIDER.with_(w_prime=30_000.0), course).time < solve(RIDER, course).time


def test_real_ride_is_rebuilt_from_checkpoint_times():
    from ttopt.real import compare

    mesh = courses.single_hill().mesh(50)
    truth = solve(RIDER, mesh, "constant")  # stands in for a real rider
    checkpoints = [4_000.0, 6_000.0, 10_000.0]
    course = courses.single_hill()
    result = compare(RIDER.with_(cp=300.0), course, checkpoints, truth.time_at(checkpoints), rebuild="constant")
    assert np.allclose(result.real.info["segment_powers"], truth.info["power"], atol=0.5)
    assert result.cyclist.cp == pytest.approx(RIDER.cp, abs=0.5)
    assert result.gain_s == pytest.approx(truth.time - solve(RIDER, mesh).time, abs=0.3)


def test_a_perfectly_paced_ride_leaves_nothing_to_gain():
    from ttopt.real import compare

    course = courses.single_hill()
    checkpoints = np.array([4_000.0, 6_000.0, 10_000.0])
    best = solve(RIDER, course.mesh(50, breaks=checkpoints[:-1]))
    result = compare(RIDER.with_(cp=300.0), course, checkpoints, best.time_at(checkpoints))
    assert result.cyclist.cp == pytest.approx(RIDER.cp, abs=0.5)
    assert abs(result.gain_s) < 0.05


def test_power_file_replay_recovers_the_drag_area():
    from ttopt.real import compare_power_file

    course = courses.rolling()
    truth_rider = RIDER.with_(cda=0.27)
    ride = solve(truth_rider, course.mesh(20), "constant")  # stands in for a recording
    # resample to a 1 Hz "power file"
    t = np.arange(0.0, ride.time, 1.0)
    dist = np.interp(t, ride.t, ride.s)
    power = np.full_like(t, ride.info["power"])
    result = compare_power_file(RIDER.with_(cda=0.2, cp=300.0), course, dist, power, t, v0=1.0, recovery=False)
    assert result.cyclist.cda == pytest.approx(0.27, abs=0.004)
    # constant power above CP never recovers, so both readings of CP agree
    assert result.cyclist.cp == pytest.approx(RIDER.cp, abs=2.0)
    assert result.real.info["cp_without_recovery"] == pytest.approx(RIDER.cp, abs=2.0)
    assert np.abs(result.replay_error_s).max() < 3.0
    assert 5.0 < result.gain_s < 20.0
    # letting the reserve refill gives the optimiser more room
    refill = compare_power_file(RIDER.with_(cda=0.2, cp=300.0), course, dist, power, t, v0=1.0)
    assert refill.gain_s > result.gain_s


def test_reserve_refills_below_critical_power_in_the_recording_model():
    from ttopt.real import critical_power_with_recovery, reserve_with_recovery

    power = np.array([500.0] * 60 + [200.0] * 600 + [500.0] * 60)
    dt = np.ones_like(power)
    w = reserve_with_recovery(power, dt, 400.0, 20_000.0)
    assert w[60] == pytest.approx(14_000.0)
    assert 14_000.0 < w[660] < 20_000.0  # partly refilled
    assert critical_power_with_recovery(power, dt, 6_000.0) == pytest.approx(400.0, abs=0.5)


def test_a_refilling_reserve_can_only_help():
    mesh = courses.rolling().mesh(50)
    strict = solve(RIDER, mesh)
    refill = solve(RIDER, mesh, recovery=True)
    assert refill.info["success"]
    assert refill.time < strict.time - 1.0
    assert refill.w_bal.min() > -1.0 and refill.w_bal.max() <= RIDER.w_prime + 1.0
    # the reserve history returned matches the recovery model applied to the plan
    from ttopt.real import reserve_with_recovery
    again = reserve_with_recovery(refill.power, refill.dt, RIDER.cp, RIDER.w_prime)
    assert np.abs(again - refill.w_bal).max() < 0.02 * RIDER.w_prime
