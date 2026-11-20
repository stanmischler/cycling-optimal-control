"""Terrain-model altitude (IGN RGE ALTI, public API, no key) at each point of the Hepburn trace."""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

folder = Path(__file__).resolve().parent.parent / "data" / "raw" / "powerfile"
data = np.genfromtxt(folder / "trace.csv", delimiter=",", names=True)
out = []
for i in range(0, len(data), 100):
    chunk = data[i : i + 100]
    query = urllib.parse.urlencode(
        {"lon": "|".join(f"{x:.6f}" for x in chunk["lon"]), "lat": "|".join(f"{x:.6f}" for x in chunk["lat"]),
         "resource": "ign_rge_alti_wld", "zonly": "true"}, safe="|")
    url = "https://data.geopf.fr/altimetrie/1.0/calcul/alti/rest/elevation.json?" + query
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "elevation-fetch/0.1 (generic research script)"})
            z = json.loads(urllib.request.urlopen(req, timeout=60).read())["elevations"]
            assert len(z) == len(chunk)
            break
        except Exception as err:
            print("retry", i, err)
            time.sleep(3 * (attempt + 1))
    else:
        raise SystemExit("failed")
    out += z
    time.sleep(0.4)
np.savetxt(folder / "elevation_ign.csv", np.column_stack([data["time_s"], out]), delimiter=",",
           header="time_s,elevation_ign_m", comments="", fmt="%.2f")
print(len(out), "points")
