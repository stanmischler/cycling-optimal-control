"""How much could each rider have gained? Scan the things we do not know.

For every rider the gain is recomputed over a grid of assumed drag area,
and corner grip. Writes data/gain_scan.json.
"""
import itertools
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from ttopt import Cyclist
from ttopt.model import Course
from ttopt.races import load_race
from ttopt.real import compare

# total mass = body mass + about 10 kg of bike, kit and bottle
RIDERS = {
    "monaco_nice_2024": {"Derek Gee": 71, "Tadej Pogacar": 65, "Jonas Vingegaard": 60, "Remco Evenepoel": 61},
    "passy_combloux_2023": {"Jonas Vingegaard": 60, "Tadej Pogacar": 65, "Wout van Aert": 78, "Neilson Powless": 66},
}
CDAS = (0.18, 0.20, 0.22, 0.24, 0.26)
GRIPS = (0.8, 1.0, 1.3)


def one(job):
    race, rider, body, cda, grip, trim = job
    course, cps, times, _ = load_race(race, lateral_g=grip)
    if trim:
        keep = course.distance >= trim
        course = Course(course.name, course.distance[keep] - trim, course.elevation[keep],
                        v_limit=course.v_limit[keep], rho=course.rho)
        cps = cps - trim
    row = dict(race=race, rider=rider, body_kg=body, cda=cda, grip=grip, trim=trim)
    try:
        cmp = compare(Cyclist(name=rider, mass=body + 10, cda=cda, crr=0.003, p_max=1000), course, cps, times[rider])
        tab = cmp.segment_table()
        row.update(gain_s=cmp.gain_s, cp=cmp.cyclist.cp, real_power=[r["real_power_W"] for r in tab],
                   optimal_power=[r["optimal_power_W"] for r in tab],
                   real_time=[r["real_time_s"] for r in tab], optimal_time=[r["optimal_time_s"] for r in tab])
    except Exception as err:  # an infeasible combination is a result too
        row.update(gain_s=None, error=str(err)[:120])
    return row


if __name__ == "__main__":
    jobs = []
    for race, riders in RIDERS.items():
        trims = (0,)
        for (rider, body), cda, grip, trim in itertools.product(riders.items(), CDAS, GRIPS, trims):
            jobs.append((race, rider, body, cda, grip, trim))
    with Pool(10) as pool:
        rows = pool.map(one, jobs, chunksize=2)
    out = Path(__file__).resolve().parent.parent / "data" / "gain_scan.json"
    out.write_text(json.dumps(rows, indent=1))
    print(len(rows), "rows ->", out)
