"""Refinement study: how accurate is each solver for a given computing time?"""

from __future__ import annotations

import time

from . import solve
from .model import Course, Cyclist

COLLOCATION_LEVELS = (400.0, 200.0, 100.0, 50.0, 25.0, 12.5)
# (interval length in m, speed step in m/s, reserve levels)
DP_LEVELS = (
    (400.0, 0.8, 6), (400.0, 0.4, 11), (200.0, 0.4, 11), (200.0, 0.2, 21),
    (100.0, 0.2, 21), (100.0, 0.1, 41), (50.0, 0.2, 21), (50.0, 0.1, 41),
    (25.0, 0.1, 41), (25.0, 0.05, 81), (12.5, 0.05, 81),
)


def refinement(
    cyclist: Cyclist,
    course: Course,
    reference_ds: float = 5.0,
    collocation_levels=COLLOCATION_LEVELS,
    dp_levels=DP_LEVELS,
    memory_limit_gb: float = 6.0,
    verbose: bool = True,
):
    """Solve at several resolutions and compare with a very fine collocation run.

    Returns (rows, reference). Each row has the method, its settings, the
    finish time, the error against the reference and the wall-clock time,
    which includes building the problem. Compilation of the DP kernels is
    excluded by a warm-up call.
    """
    reference = solve(cyclist, course, "collocation", ds=reference_ds)
    solve(cyclist, course, "dp", ds=max(course.length / 20, 100.0), dv=0.5, n_w=5)  # warm-up
    rows = []

    def record(method, settings, **kw):
        tic = time.perf_counter()
        try:
            plan = solve(cyclist, course, method, **kw)
        except MemoryError as err:
            if verbose:
                print(f"  skipped {method} {settings}: {err}")
            return
        wall = time.perf_counter() - tic
        row = {
            "method": method, **settings, "time_s": plan.time,
            "error_s": plan.time - reference.time, "seconds": wall,
            "intervals": plan.mesh.n,
        }
        rows.append(row)
        if verbose:
            print(f"  {method:<12s} {str(settings):<42s} T={plan.time:10.3f}  err={row['error_s']:+8.3f} s  in {wall:7.3f} s")

    for ds in collocation_levels:
        record("collocation", {"ds": ds}, ds=ds)
    for ds, dv, n_w in dp_levels:
        record("dp", {"ds": ds, "dv": dv, "n_w": n_w}, ds=ds, dv=dv, n_w=n_w, memory_limit_gb=memory_limit_gb)
    return rows, reference
