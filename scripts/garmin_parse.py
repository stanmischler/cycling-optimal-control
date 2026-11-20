"""Turn the Garmin .fit files in data/raw/garmin/fit into one CSV per activity."""
import csv
import sys
from pathlib import Path

import fitdecode

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "garmin" / "fit"
OUT = ROOT / "data" / "raw" / "garmin" / "csv"
FIELDS = ["timestamp", "position_lat", "position_long", "distance", "enhanced_altitude", "enhanced_speed",
          "heart_rate", "power", "cadence", "temperature"]


def parse(path):
    rows, sessions = [], []
    with fitdecode.FitReader(str(path)) as fit:
        for frame in fit:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            if frame.name == "record":
                rows.append([frame.get_value(f, fallback=None) for f in FIELDS])
            elif frame.name == "session":
                sessions.append({k: frame.get_value(k, fallback=None) for k in
                                 ("sport", "sub_sport", "start_time", "total_timer_time", "total_distance",
                                  "avg_power", "avg_heart_rate", "threshold_power")})
    return rows, sessions


def main():
    OUT.mkdir(exist_ok=True)
    for folder in sorted(SRC.iterdir()):
        for path in folder.glob("*.fit"):
            rows, sessions = parse(path)
            semi = 180.0 / 2**31
            with open(OUT / f"{folder.name}.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["t", "lat", "lon", "distance_m", "altitude_m", "speed_ms", "hr", "power_w", "cadence", "temp_c"])
                t0 = rows[0][0]
                for r in rows:
                    w.writerow([(r[0] - t0).total_seconds(), r[1] * semi if r[1] is not None else "",
                                r[2] * semi if r[2] is not None else ""] + ["" if v is None else v for v in r[3:]])
            n_p = sum(1 for r in rows if r[7] is not None)
            n_h = sum(1 for r in rows if r[6] is not None)
            print(folder.name, len(rows), "records, power", n_p, "hr", n_h, "|",
                  [(s["sport"], str(s["start_time"])[:16], round(s["total_timer_time"] or 0), round(s["total_distance"] or 0),
                    s["avg_power"], s["avg_heart_rate"]) for s in sessions])


if __name__ == "__main__":
    main()
