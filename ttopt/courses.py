"""Courses: synthetic profiles for development and loaders for real tracks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .model import G, Course


def _ramp(x, start, end):
    """Smooth 0 -> 1 transition between start and end (C1)."""
    u = np.clip((x - start) / (end - start), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def flat(length: float = 10_000.0, name: str = "flat") -> Course:
    x = np.linspace(0.0, length, int(length / 10) + 1)
    return Course(name, x, np.zeros_like(x))


def single_hill(
    length: float = 10_000.0,
    climb_start: float = 4_000.0,
    climb_length: float = 2_000.0,
    gradient: float = 0.07,
    name: str = "single hill",
) -> Course:
    """Flat, then a climb at ``gradient``, then flat at the top."""
    x = np.linspace(0.0, length, int(length / 10) + 1)
    blend = 100.0  # metres over which the gradient changes
    g = gradient * (
        _ramp(x, climb_start - blend / 2, climb_start + blend / 2)
        - _ramp(x, climb_start + climb_length - blend / 2, climb_start + climb_length + blend / 2)
    )
    h = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(x))])
    return Course(name, x, h)


def rolling(
    length: float = 20_000.0,
    wavelength: float = 4_000.0,
    amplitude: float = 40.0,
    name: str = "rolling",
) -> Course:
    x = np.linspace(0.0, length, int(length / 10) + 1)
    h = amplitude * (1.0 - np.cos(2.0 * np.pi * x / wavelength))
    return Course(name, x, h)


def climb_and_descent(
    length: float = 12_000.0,
    gradient_up: float = 0.06,
    gradient_down: float = 0.06,
    flat_each_end: float = 2_000.0,
    name: str = "climb and descent",
) -> Course:
    """Flat, up, straight back down, flat. Finish at the start altitude."""
    x = np.linspace(0.0, length, int(length / 10) + 1)
    hill = length - 2.0 * flat_each_end
    up = hill * gradient_down / (gradient_up + gradient_down)
    top = flat_each_end + up
    blend = 100.0
    g = (
        gradient_up * (_ramp(x, flat_each_end - blend / 2, flat_each_end + blend / 2) - _ramp(x, top - blend / 2, top + blend / 2))
        - gradient_down * (_ramp(x, top - blend / 2, top + blend / 2) - _ramp(x, length - flat_each_end - blend / 2, length - flat_each_end + blend / 2))
    )
    h = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(x))])
    return Course(name, x, h)


SYNTHETIC = {
    "flat": flat,
    "single hill": single_hill,
    "rolling": rolling,
    "climb and descent": climb_and_descent,
}


# --------------------------------------------------------------------------
# Real tracks
# --------------------------------------------------------------------------


def _haversine(lat1, lon1, lat2, lon2):
    r = 6_371_008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def _gaussian_smooth(y, step, sigma):
    """Gaussian filter with reflected ends; ``sigma`` and ``step`` in metres."""
    if sigma <= 0:
        return y.copy()
    half = int(np.ceil(4 * sigma / step))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) * step / sigma) ** 2)
    k /= k.sum()
    padded = np.concatenate([2 * y[0] - y[half:0:-1], y, 2 * y[-1] - y[-2 : -half - 2 : -1]])
    return np.convolve(padded, k, mode="valid")


def corner_speed_limit(lat, lon, distance, step, lateral_g=1.0, sigma=40.0, v_cap=27.0, v_floor=8.0):
    """Speed limit (m/s) from the curvature of the track.

    v = sqrt(a_lat * R), with the path smoothed over ``sigma`` metres first so
    that jitter in the track does not create fake corners. ``v_floor`` is the
    slowest any corner is allowed to be: a drawn centreline makes hairpins
    much tighter than the line a rider takes through them. Without the floor
    the model brakes for every hairpin of a climb, which riders do not.
    """
    r = 6_371_008.8
    lat0 = np.radians(lat.mean())
    x = np.radians(lon - lon[0]) * r * np.cos(lat0)
    y = np.radians(lat - lat[0]) * r
    x, y = _gaussian_smooth(x, step, sigma), _gaussian_smooth(y, step, sigma)
    dx, dy = np.gradient(x, distance), np.gradient(y, distance)
    ddx, ddy = np.gradient(dx, distance), np.gradient(dy, distance)
    curvature = np.abs(dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-9)
    with np.errstate(divide="ignore"):
        v = np.sqrt(lateral_g * G / np.maximum(curvature, 1e-9))
    return np.clip(v, v_floor, v_cap)


def _median_filter(y, half):
    if half <= 0:
        return y.copy()
    padded = np.concatenate([np.full(half, y[0]), y, np.full(half, y[-1])])
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    return np.median(windows, axis=1)


def from_track(
    name: str,
    lat,
    lon,
    elevation,
    step: float = 10.0,
    smooth_elevation: float = 60.0,
    despike: float = 80.0,
    bad_elevation=None,
    corners: bool = True,
    lateral_g: float = 1.0,
    v_cap: float = 27.0,
    **course_kwargs,
) -> Course:
    """Build a course from raw GPS points.

    The track is resampled every ``step`` metres, the elevation is smoothed
    with a Gaussian of ``smooth_elevation`` metres (raw GPS/DEM altitude is too
    noisy to differentiate), and corner speed limits are derived from the
    path curvature when ``corners`` is true.

    ``bad_elevation`` marks points whose altitude is wrong (tunnels and
    bridges in a terrain model); they are bridged by linear interpolation.
    ``despike`` is the width in metres of a median filter applied before the
    smoothing, which removes isolated altitude spikes.
    """
    lat, lon, ele = (np.asarray(a, dtype=float) for a in (lat, lon, elevation))
    bad = np.zeros(len(ele), bool) if bad_elevation is None else np.asarray(bad_elevation, bool)
    hop = _haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
    keep = np.concatenate([[True], hop > 0.5])  # drop duplicate points
    lat, lon, ele, bad = lat[keep], lon[keep], ele[keep], bad[keep]
    hop = _haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
    flat_d = np.concatenate([[0.0], np.cumsum(hop)])
    if bad.any():
        ele = np.interp(flat_d, flat_d[~bad], ele[~bad])

    n = int(round(flat_d[-1] / step))
    d = np.linspace(0.0, flat_d[-1], n + 1)
    stepd = d[1] - d[0]
    lat_r, lon_r = np.interp(d, flat_d, lat), np.interp(d, flat_d, lon)
    ele_r = _median_filter(np.interp(d, flat_d, ele), int(round(0.5 * despike / stepd)))
    ele_r = _gaussian_smooth(ele_r, stepd, smooth_elevation)

    # road distance includes the vertical component
    road = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(d), np.diff(ele_r)))])
    v_limit = corner_speed_limit(lat_r, lon_r, d, stepd, lateral_g, v_cap=v_cap) if corners else None
    course = Course(name, road, ele_r, v_limit=v_limit, lat=lat_r, lon=lon_r, **course_kwargs)
    course.meta["map_distance"] = d  # horizontal distance, as printed on race maps
    return course


def from_gpx(path: str | Path, name: str | None = None, **kwargs) -> Course:
    import gpxpy

    with open(path, "r", encoding="utf-8") as fh:
        gpx = gpxpy.parse(fh)
    pts = [p for trk in gpx.tracks for seg in trk.segments for p in seg.points]
    if not pts:
        pts = [p for rte in gpx.routes for p in rte.points]
    if not pts:
        raise ValueError(f"no track or route points in {path}")
    if any(p.elevation is None for p in pts):
        raise ValueError(f"{path} has points without elevation")
    return from_track(
        name or Path(path).stem,
        [p.latitude for p in pts],
        [p.longitude for p in pts],
        [p.elevation for p in pts],
        **kwargs,
    )


def from_csv(path: str | Path, name: str | None = None, **kwargs) -> Course:
    """CSV with lat, lon and elevation_m columns; optional tunnel / bridge flags."""
    data = np.genfromtxt(path, delimiter=",", names=True, usecols=range(9))
    cols = {k.lower(): k for k in data.dtype.names}
    ele = cols.get("elevation_m") or cols.get("elevation") or cols.get("ele")
    bad = np.zeros(len(data), bool)
    for flag in ("tunnel", "bridge"):
        if flag in cols:
            bad |= data[cols[flag]] > 0
    kwargs.setdefault("bad_elevation", bad)
    return from_track(name or Path(path).stem, data[cols["lat"]], data[cols["lon"]], data[ele], **kwargs)
