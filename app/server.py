"""Pacing app: a local web page in front of the ttopt solver.

    .venv/bin/python app/server.py            # opens http://127.0.0.1:8309
    .venv/bin/python app/server.py --port 9000 --no-browser

The server only listens on this machine. Uploaded GPX files are parsed in
memory and never written to disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ttopt import Cyclist, courses, solve  # noqa: E402
from ttopt.model import Course  # noqa: E402
from ttopt.races import RACES, load_race  # noqa: E402

PAGE = Path(__file__).resolve().parent / "index.html"
MAX_UPLOAD = 40 * 1024 * 1024
SOLVE_LOCK = threading.Lock()

BUILTIN = {
    "monaco_nice_2024": "Monaco → Nice (Tour de France 2024, stage 21)",
    "passy_combloux_2023": "Passy → Combloux (Tour de France 2023, stage 16)",
    "single hill": "Test course: 10 km with a 2 km climb at 7 %",
    "rolling": "Test course: 20 km of rolling hills",
    "climb and descent": "Test course: 12 km, up and straight back down",
    "flat": "Test course: 10 km flat",
}

GRADE_EDGES = np.array([-6.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0])  # per cent
GRADE_LABELS = ["< −6", "−6 to −3", "−3 to −1", "−1 to 1", "1 to 3", "3 to 5", "5 to 7", "7 to 9", "> 9"]


class UserError(ValueError):
    """A problem the person can fix; its message is shown in the page."""


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def _number(data: dict, key: str, lo: float, hi: float, label: str) -> float:
    try:
        x = float(data[key])
    except (KeyError, TypeError, ValueError):
        raise UserError(f"{label} is missing or not a number.")
    if not np.isfinite(x) or not lo <= x <= hi:
        raise UserError(f"{label} must be between {lo:g} and {hi:g}.")
    return x


def build_cyclist(r: dict) -> Cyclist:
    body = _number(r, "body_kg", 30, 200, "Body weight")
    bike = _number(r, "bike_kg", 3, 40, "Bike weight")
    cp = _number(r, "cp", 50, 600, "Critical power")
    w_prime = _number(r, "w_prime_kj", 0, 60, "Reserve") * 1000.0
    p_max = _number(r, "p_max", 60, 2500, "Maximum power")
    if p_max < cp:
        raise UserError("Maximum power must be at least your critical power.")
    return Cyclist(
        name=str(r.get("name") or "you")[:60],
        mass=body + bike,
        cda=_number(r, "cda", 0.12, 0.8, "Drag area"),
        crr=_number(r, "crr", 0.001, 0.03, "Rolling resistance"),
        cp=cp,
        w_prime=w_prime,
        p_max=p_max,
    )


def course_from_gpx(text: str, name: str, lateral_g: float, v_cap: float) -> Course:
    import gpxpy

    try:
        gpx = gpxpy.parse(text)
    except Exception:
        raise UserError("This file could not be read as GPX.")
    pts = [p for trk in gpx.tracks for seg in trk.segments for p in seg.points]
    if not pts:
        pts = [p for rte in gpx.routes for p in rte.points]
    if len(pts) < 10:
        raise UserError("The GPX file has no track (fewer than 10 points).")
    if any(p.elevation is None for p in pts):
        raise UserError(
            "The GPX file has no elevation. Export it again with elevation "
            "(Strava, Komoot and RideWithGPS routes include it)."
        )
    title = name or next((t.name for t in gpx.tracks if t.name), None) or gpx.name or "Your course"
    course = courses.from_track(
        str(title)[:80],
        [p.latitude for p in pts],
        [p.longitude for p in pts],
        [p.elevation for p in pts],
        lateral_g=lateral_g,
        v_cap=v_cap,
    )
    if course.length < 500:
        raise UserError("The course is shorter than 500 m.")
    if course.length > 400_000:
        raise UserError("The course is longer than 400 km.")
    return course


def build_course(c: dict) -> Course:
    lateral_g = _number(c, "lateral_g", 0.2, 1.3, "Cornering")
    v_cap = _number(c, "v_max_kmh", 25, 110, "Top speed") / 3.6
    if c.get("gpx"):
        course = course_from_gpx(str(c["gpx"]), str(c.get("name") or ""), lateral_g, v_cap)
    else:
        key = c.get("builtin")
        if key in RACES:
            course = load_race(key, lateral_g=lateral_g, v_cap=v_cap)[0]
        elif key in courses.SYNTHETIC:
            course = courses.SYNTHETIC[key]()
            course.v_limit = np.full(course.distance.shape, v_cap)
        else:
            raise UserError("Choose a course or drop a GPX file.")
        course.name = BUILTIN[key]

    # air density from temperature and the mean altitude of the course
    temp = _number(c, "temperature_c", -15, 50, "Temperature")
    altitude = float(np.mean(course.elevation)) if course.lat is not None else 0.0
    course.rho = 1.225 * np.exp(-max(altitude, 0.0) / 8400.0) * 288.15 / (273.15 + temp)

    # wind: with a map the direction matters, without one it is a pure headwind
    wind = _number(c, "wind_kmh", 0, 80, "Wind speed") / 3.6
    if wind > 0.0:
        if course.lat is not None:
            lat, lon = np.radians(course.lat), np.radians(course.lon)
            east = np.gradient(lon) * np.cos(lat)
            north = np.gradient(lat)
            k = np.ones(21) / 21.0  # steady the heading over about 200 m
            east, north = np.convolve(east, k, "same"), np.convolve(north, k, "same")
            heading = np.arctan2(east, north)
            wind_from = np.radians(_number(c, "wind_from_deg", 0, 360, "Wind direction"))
            course.headwind = wind * np.cos(heading - wind_from)
        else:
            course.headwind = wind
    return course


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------


def _smooth_grade(s, h, half=100.0):
    mid = 0.5 * (s[:-1] + s[1:])
    a, b = np.clip(mid - half, 0, s[-1]), np.clip(mid + half, 0, s[-1])
    return (np.interp(b, s, h) - np.interp(a, s, h)) / np.maximum(b - a, 1e-6)


def cue_segments(s, h, max_segments=22):
    """Cut the course into stretches of similar gradient: [(i0, i1), ...]."""
    grade = _smooth_grade(s, h) * 100.0
    band = np.digitize(grade, GRADE_EDGES)
    cuts = [0] + [i for i in range(1, len(band)) if band[i] != band[i - 1]] + [len(band)]
    runs = [[cuts[j], cuts[j + 1]] for j in range(len(cuts) - 1)]
    length = s[-1]
    min_len = max(300.0, length / 40.0)

    def run_len(r):
        return s[r[1]] - s[r[0]]

    def run_grade(r):
        return (h[r[1]] - h[r[0]]) / run_len(r)

    while len(runs) > 1:
        j = min(range(len(runs)), key=lambda q: run_len(runs[q]))
        if run_len(runs[j]) >= min_len and len(runs) <= max_segments:
            break
        sides = [q for q in (j - 1, j + 1) if 0 <= q < len(runs)]
        q = min(sides, key=lambda q: abs(run_grade(runs[q]) - run_grade(runs[j])))
        lo, hi = min(j, q), max(j, q)
        runs[lo:hi + 1] = [[runs[lo][0], runs[hi][1]]]

    # neighbours that ended up with nearly the same gradient are one stretch
    j = 0
    while j < len(runs) - 1:
        if abs(run_grade(runs[j]) - run_grade(runs[j + 1])) < 0.01:
            runs[j:j + 2] = [[runs[j][0], runs[j + 1][1]]]
        else:
            j += 1

    # long uniform stretches are split so that there is a target every few km
    longest = max(5000.0, length / 8.0)
    out = []
    for i0, i1 in runs:
        pieces = int(np.ceil((s[i1] - s[i0]) / longest))
        edges = np.searchsorted(s, np.linspace(s[i0], s[i1], pieces + 1))
        edges[0], edges[-1] = i0, i1
        out += [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    return out


def describe(plan, base, course: Course) -> dict:
    m, c = plan.mesh, plan.cyclist
    s, h, t = m.s, m.h, plan.t
    dt = plan.dt
    grade = _smooth_grade(s, h) * 100.0

    segments = []
    for i0, i1 in cue_segments(s, h):
        seg_t = t[i1] - t[i0]
        length = s[i1] - s[i0]
        segments.append({
            "from_km": s[i0] / 1000.0, "to_km": s[i1] / 1000.0,
            "grade": 100.0 * (h[i1] - h[i0]) / length,
            "power": float((plan.power[i0:i1] * dt[i0:i1]).sum() / seg_t),
            "power_constant": float((base.power[i0:i1] * base.dt[i0:i1]).sum() / (base.t[i1] - base.t[i0])),
            "speed_kmh": 3.6 * length / seg_t,
            "time": seg_t, "elapsed": t[i1],
        })

    band = np.digitize(grade, GRADE_EDGES)
    bands = []
    for b, label in enumerate(GRADE_LABELS):
        pick = band == b
        if dt[pick].sum() < 5.0:
            continue
        bands.append({
            "label": label,
            "power": float((plan.power[pick] * dt[pick]).sum() / dt[pick].sum()),
            "speed_kmh": float(3.6 * m.ds[pick].sum() / dt[pick].sum()),
            "time": float(dt[pick].sum()),
            "km": float(m.ds[pick].sum() / 1000.0),
        })

    track = None
    if course.lat is not None:
        track = {
            "lat": np.interp(s, course.distance, course.lat),
            "lon": np.interp(s, course.distance, course.lon),
        }

    warnings = []
    if not plan.info.get("success", True):
        warnings.append(
            f"The optimiser stopped early ({plan.info.get('status')}); treat this plan as approximate."
        )
    if plan.time > 75 * 60:
        warnings.append(
            "This effort lasts well over an hour. Critical power is only sustainable for "
            "30–60 minutes: enter the power you can hold for this duration instead, "
            "or the plan will be too hard to follow."
        )
    if c.w_prime > 0 and plan.w_bal[-1] > 0.05 * c.w_prime:
        warnings.append("The plan does not use the whole reserve: the power cap or the speed limits are in the way.")

    return {
        "course": {
            "name": course.name, "km": s[-1] / 1000.0, "ascent": course.total_ascent,
            "rho": m.rho, "has_map": track is not None,
        },
        "rider": {"mass": c.mass, "cp": c.cp, "w_prime": c.w_prime, "p_max": c.p_max, "cda": c.cda, "crr": c.crr},
        "summary": {
            "time": plan.time, "time_constant": base.time, "gain": base.time - plan.time,
            "mean_power": plan.mean_power, "constant_power": float(base.info.get("power", base.mean_power)),
            "mean_speed_kmh": plan.mean_speed_kmh, "max_speed_kmh": float(plan.v.max() * 3.6),
            "work_kj": float((plan.power * dt).sum() / 1000.0),
            "solve_seconds": plan.solve_seconds, "intervals": int(m.n),
        },
        "trace": {
            "s": s, "h": h, "t": t, "grade": grade,
            "power": plan.power, "v": plan.v * 3.6, "reserve": np.maximum(plan.w_bal, 0.0) / 1000.0,
            "power_constant": base.power, "v_constant": base.v * 3.6, "t_constant": base.t,
            "wind": m.wind * 3.6,
        },
        "track": track,
        "segments": segments,
        "bands": bands,
        "warnings": warnings,
    }


def run(request: dict) -> dict:
    cyclist = build_cyclist(request.get("rider") or {})
    course = build_course(request.get("course") or {})
    ds = float(np.clip(course.length / 1600.0, 25.0, 250.0))
    mesh = course.mesh(ds)
    with SOLVE_LOCK:
        base = solve(cyclist, mesh, method="constant")
        plan = solve(cyclist, mesh, method="collocation")
    return describe(plan, base, course)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return [round(float(v), 5 if abs(float(v)) < 200 else 2) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    raise TypeError(type(x))


class Handler(BaseHTTPRequestHandler):
    server_version = "ttopt"

    def _send(self, code: int, body: bytes, kind: str):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict):
        self._send(code, json.dumps(payload, default=_jsonable).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/courses":
            self._json(200, {"courses": [{"key": k, "label": v} for k, v in BUILTIN.items()]})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/solve":
            return self._send(404, b"not found", "text/plain")
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size > MAX_UPLOAD:
                raise UserError("The file is larger than 40 MB.")
            request = json.loads(self.rfile.read(size))
            self._json(200, run(request))
        except UserError as err:
            self._json(400, {"error": str(err)})
        except Exception as err:  # shown in the page; the trace goes to the terminal
            traceback.print_exc()
            self._json(500, {"error": f"The solver failed: {err}"})

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s\n" % (fmt % args))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8309)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    # compile the simulation kernels now rather than on the first request
    threading.Thread(
        target=lambda: solve(Cyclist(), courses.flat(2000.0), method="constant"), daemon=True
    ).start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Pacing app running at {url}  (Ctrl-C to stop)")
    if not args.no_browser:
        threading.Timer(0.6, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
