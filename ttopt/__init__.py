"""ttopt: optimal pacing for a cycling time trial.

    from ttopt import Cyclist, courses, solve
    plan = solve(Cyclist(), courses.single_hill(), method="collocation")
    print(plan.summary())
"""

from __future__ import annotations

from . import courses
from .collocation import solve_collocation
from .dp import solve_dp
from .model import Course, Cyclist, Mesh, Strategy, constant_power, simulate

__all__ = [
    "Course", "Cyclist", "Mesh", "Strategy", "courses",
    "solve", "solve_collocation", "solve_dp", "simulate", "constant_power",
]


def solve(
    cyclist: Cyclist,
    course: Course | Mesh,
    method: str = "collocation",
    ds: float = 50.0,
    v0: float = 1.0,
    **options,
) -> Strategy:
    """Fastest pacing plan for ``cyclist`` on ``course``.

    method   "collocation" (fast, accurate, one trajectory),
             "dp" (dynamic programming: global on its grid, gives a feedback law),
             "constant" (baseline: constant power that empties the reserve)
    ds       length of the course intervals in metres
    v0       speed when the clock starts (m/s)
    options  passed to the solver (e.g. ``dv``, ``n_w`` for dp; ``tol`` for collocation)
    """
    mesh = course if isinstance(course, Mesh) else course.mesh(ds)
    if method == "collocation":
        return solve_collocation(cyclist, mesh, v0=v0, **options)
    if method == "dp":
        return solve_dp(cyclist, mesh, v0=v0, **options)
    if method == "constant":
        return constant_power(cyclist, mesh, v0=v0)
    raise ValueError(f"unknown method {method!r}")
