"""Michael Hepburn, Tour de France 2018 stage 20 (Saint-Pee-sur-Nivelle - Espelette).

A real pro time trial with the rider's own 1 Hz power file. The file is
replayed through the model, then the optimiser plans the same rider's best ride.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from ttopt import Cyclist, courses, solve
from ttopt.courses import _haversine
from ttopt.real import compare_power_file, speed_limits_from_recording

ROOT = Path(__file__).resolve().parent.parent
TRACE = ROOT / "data" / "raw" / "powerfile" / "trace.csv"

# 77 kg rider (his account weight) + about 10 kg of bike, kit and bottle
HEPBURN = Cyclist(name="Hepburn", mass=87.0, cda=0.22, crr=0.0035, w_prime=20_000.0, p_max=1000.0)
RHO = 1.17  # about 25 C near sea level


def load(smooth_elevation=40.0, elevation="ign"):
    """Returns (course, sample position on the course, speed, samples).

    elevation  "ign": French terrain model sampled at his GPS points. It
               reproduces his real clock twice as well as the head unit's
               barometric altitude ("baro"), which has steps of 20-30 m.
    """
    data = np.genfromtxt(TRACE, delimiter=",", names=True)
    if elevation == "ign":
        height = np.genfromtxt(TRACE.parent / "elevation_ign.csv", delimiter=",", names=True)["elevation_ign_m"]
    else:
        height = data["elevation_m"]
    course = courses.from_track(
        "Saint-Pee-sur-Nivelle - Espelette, Tour de France 2018 stage 20",
        data["lat"], data["lon"], height,
        smooth_elevation=smooth_elevation, despike=0.0, corners=False, rho=RHO,
    )
    hop = _haversine(data["lat"][:-1], data["lon"][:-1], data["lat"][1:], data["lon"][1:])
    flat = np.concatenate([[0.0], np.cumsum(hop)])
    position = np.interp(flat, course.meta["map_distance"], course.distance)
    speed = np.gradient(position, data["time_s"])
    course.v_limit = speed_limits_from_recording(course, position, speed, data["power_w"])
    return course, position, speed, data


def heading_unit_vectors(course):
    """East and north components of the direction of travel along the course."""
    lat, lon = course.lat, course.lon
    east = np.gradient(lon) * np.cos(np.radians(lat.mean()))
    north = np.gradient(lat)
    norm = np.maximum(np.hypot(east, north), 1e-12)
    return east / norm, north / norm


def fit_drag_and_wind(rider, course, position, data, speed, ds=25.0):
    """Drag area and a steady wind that best reproduce his clock along the course.

    Three numbers are fitted to about 1,200 clock readings: the drag area and
    the two components of a constant wind. The headwind at each point follows
    from the direction of travel.
    """
    from scipy.optimize import least_squares

    from ttopt.model import simulate
    from ttopt.real import power_on_mesh

    east, north = heading_unit_vectors(course)
    mesh0 = course.mesh(ds)
    power, t_nodes = power_on_mesh(mesh0, position, data["power_w"], data["time_s"])
    wide = rider.with_(p_max=2000.0, f_max=2000.0)

    def clock_error(x):
        cda, wx, wy = x
        course.headwind = -(wx * east + wy * north)  # wind blowing towards (wx, wy)
        ride = simulate(wide.with_(cda=cda), course.mesh(ds), power, float(speed[0]))
        return ride.t - (t_nodes - t_nodes[0])

    best = least_squares(clock_error, x0=[0.2, 0.0, 0.0], bounds=([0.1, -12, -12], [0.5, 12, 12]), diff_step=1e-3)
    cda, wx, wy = best.x
    course.headwind = -(wx * east + wy * north)
    return cda, wx, wy


def run(rider=HEPBURN, ds=25.0, wind=False, recovery=True, **load_options):
    course, position, speed, data = load(**load_options)
    info = {}
    if wind:
        cda, wx, wy = fit_drag_and_wind(rider, course, position, data, speed, ds)
        rider = rider.with_(cda=cda)
        info = {"wind_speed_ms": float(np.hypot(wx, wy)),
                "wind_from_deg": float((np.degrees(np.arctan2(-wx, -wy))) % 360.0)}
    result = compare_power_file(rider, course, position, data["power_w"], data["time_s"],
                                v0=float(speed[0]), ds=ds, fit=None if wind else "cda", recovery=recovery)
    result.real.info.update(info)
    return course, position, speed, data, result


def normalised_power(power_1hz):
    """Standard normalised power: fourth-power mean of the 30 s rolling average."""
    rolling = np.convolve(power_1hz, np.ones(30) / 30, "valid")
    return float((rolling**4).mean() ** 0.25)


def three_way(rider=HEPBURN, ds=25.0, **load_options):
    """His real ride against the optimum under three ways of fixing 'same effort'."""
    course, position, speed, data, strict = run(rider, ds, recovery=False, **load_options)
    real, fitted, v0 = strict.real, strict.cyclist, float(speed[0])
    np_real = normalised_power(data["power_w"])
    capped = fitted.with_(p_max=float(real.power.max()))
    plans = {
        "no refill": strict.optimal,
        "equal normalised power": solve(capped, real.mesh, v0=v0, effort_cap=np_real),
        "reserve refills": solve(fitted, real.mesh, v0=v0, recovery=True),
    }
    return dict(course=course, position=position, speed=speed, data=data, real=real, rider=fitted,
                plans=plans, np_real=np_real, clock_error=strict.replay_error_s)


if __name__ == "__main__":
    out = three_way()
    real, r = out["real"], out["rider"]
    err = out["clock_error"]
    print(f"course {out['course'].length/1000:.2f} km, {out['course'].total_ascent:.0f} m of climbing")
    print(f"fitted drag area {r.cda:.3f} m2; critical power {r.cp:.0f} W; "
          f"recorded mean {out['data']['power_w'].mean():.0f} W, normalised {out['np_real']:.0f} W")
    print(f"model fed his power: clock error rms {np.sqrt((err**2).mean()):.1f} s, max {np.abs(err).max():.1f} s")
    print(f"real ride {real.time:.1f} s")
    for name, plan in out["plans"].items():
        resampled = np.interp(np.arange(0, plan.time), plan.t[:-1], plan.power)
        print(f"  optimum, {name:<24s} {plan.time:7.1f} s  gain {real.time - plan.time:+6.1f} s  "
              f"mean {plan.mean_power:4.0f} W  normalised {normalised_power(resampled):4.0f} W  ({plan.info['status']})")
