"""The two real time trials: course, checkpoints and official times."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import courses
from .model import Course

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"

# Where each checkpoint sits on our track, in map kilometres.
# 2024: the track is 0.14 km longer than the official 33.7 km (uncertain
# U-turn in Monaco), the two summits are located on the track itself, and the
# third check is the corner of Place Ile de Beaute on the track.
# 2023: official kilometres, stretched by the 0.16 % length difference.
RACES = {
    "monaco_nice_2024": {
        "title": "Monaco - Nice, Tour de France 2024 stage 21",
        "points": "monaco_nice_2024_points.csv",
        "checkpoints_km": {"T1": 11.20, "T2": 17.15, "T3": 28.87, "finish": None},
        "labels": {"T1": "La Turbie", "T2": "Col d'Eze", "T3": "Nice", "finish": "Finish"},
        "rho": 1.15,  # about 27 C near sea level, thinner on the climbs
    },
    "passy_combloux_2023": {
        "title": "Passy - Combloux, Tour de France 2023 stage 16",
        "points": "passy_combloux_2023_points.csv",
        "checkpoints_km": {"T1": 7.1 * 22.436 / 22.4, "T2": 16.1 * 22.436 / 22.4, "T3": 18.9 * 22.436 / 22.4, "finish": None},
        "labels": {"T1": "Passy Chef-Lieu", "T2": "Domancy", "T3": "Cote de Domancy", "finish": "Finish"},
        "rho": 1.07,  # about 31 C at 550-970 m
    },
}


def load_race(key: str, **track_options):
    """Returns (course, checkpoints in road metres, {rider: times in s}, labels)."""
    spec = RACES[key]
    course: Course = courses.from_csv(DATA / spec["points"], name=spec["title"], rho=spec["rho"], **track_options)
    map_d = course.meta["map_distance"]
    ids = list(spec["checkpoints_km"])
    road = np.array([
        course.length if km is None else np.interp(km * 1000.0, map_d, course.distance)
        for km in spec["checkpoints_km"].values()
    ])
    with open(DATA / "splits.json", encoding="utf-8") as fh:
        riders = json.load(fh)[key]["riders"]
    times = {
        name: np.array([r["times_s"][i] for i in ids], dtype=float)
        for name, r in riders.items()
        if all(r["times_s"].get(i) is not None for i in ids)
    }
    return course, road, times, [spec["labels"][i] for i in ids]


def map_km_to_road(course: Course, km: float) -> float:
    return float(np.interp(km * 1000.0, course.meta["map_distance"], course.distance))
