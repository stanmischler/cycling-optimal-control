Provenance of the GPS tracks in this folder (fetched 2026-11-20). Nothing here is smoothed or resampled.

FILES
  passy_combloux_2023.gpx                   verbatim download (sha256 5c63ef96...31231d)
      https://cdn.cyclingstage.com/images/tour-de-france/2023/stage-16-parcours.gpx
      linked from https://www.cyclingstage.com/tour-de-france-2023-gpx/ (free public download, no login)
      1860 trkpts, integer-metre elevations (coarse DEM, up to ~29 m off the IGN DEM).
  passy_combloux_2023_points.csv            same 1860 points, same order, plus IGN elevation (see CSV COLUMNS)
  monaco_nice_2024_cyclingstage_original.gpx verbatim download (sha256 13d736f2...ca8b486)
      https://cdn.cyclingstage.com/images/tour-de-france/2024/stage-21-route.gpx
      linked from https://www.cyclingstage.com/tour-de-france-2024-gpx/
      35.16 km, NOT the raced 33.7 km route in two places (see below). Kept for reference only.
  monaco_nice_2024.gpx                      REBUILT track, 3248 trkpts, <ele> = IGN RGE ALTI (not the cyclingstage value)
  monaco_nice_2024_points.csv               same 3248 points, same order (see CSV COLUMNS)

CSV COLUMNS
  idx, lat, lon,
  elevation_m    IGN RGE ALTI sampled at the exact lat/lon (Geoplateforme altimetry REST API,
                 https://data.geopf.fr/altimetrie/1.0/calcul/alti/rest/elevation.json, resource ign_rge_alti_wld,
                 100 points per request, 0.4 s pause). It is a bare-ground terrain model: wrong in tunnels and on bridges.
  distance_m     cumulative haversine distance along the points (2D)
  ele_source_m   elevation printed in the cyclingstage GPX (blank for the inserted OSM points)
  origin         cyclingstage | osm_quays
  tunnel, bridge 0/1 from OpenStreetMap, via one map-matching request to the public Valhalla server
                 (valhalla1.openstreetmap.de /trace_attributes, pedestrian costing)
  road           OSM road name/ref the point was matched to

HOW monaco_nice_2024.gpx WAS REBUILT
  Road sequence of the cyclingstage track was checked against the official roadbook itinerary
  (ASO roadbook 2024, pages 204-210, copy at https://ciclo21.com/wp-content/uploads/2024/06/Tour-Francia-Libro-ruta-2024.pdf):
  Bd Albert Ier - Bd Louis II (tunnel) - Av. Princesse Grace out and back - Portier - Av. des Spelugues - Av. de la Madone -
  Bd Princesse Charlotte - Beausoleil D53 - D6007 - D53 - La Turbie D2564 - M45 down to Eze - M6007 - M46 up to Col d'Eze -
  M2564 - Col des Quatre Chemins - M33 - M6007 (Corne d'Or, Corniche Andre de Joly) - Rue Barla - Rue Arson -
  Place Ile de Beaute - quays - Promenade des Anglais - U-turn - Promenade - Av. Max Gallo - Place Massena.
  The cyclingstage track follows these roads except in two places, which were spliced:
  1. Monaco U-turn (INFERRED, not confirmed by a street-level source). cyclingstage turns at the far east end of
     Av. Princesse Grace (43.74935,7.43849). That puts every later official km mark 0.75-0.95 km late. The turn was moved
     to the median crossover at 43.74769,7.43482 (OSM way 93137593): cyclingstage points 192..299 dropped, point 191 joined
     to point 300 (11.7 m apart). Removes 754 m. With this, Beausoleil (74 m) falls at km 3.86 vs official 3.8/3.9.
     The next crossover west (43.74675,7.43331) would give a 33.54 km total and Beausoleil at km 3.56. Uncertainty of the
     turn position: about +/-75 m, i.e. +/-150 m of course length, all within the flat first 3 km.
  2. Nice port. cyclingstage goes Rue Cassini - Rue Segurane - Rue de Foresta (a one-way-legal detour). The official
     last-kilometres map shows Quai Papacino - Quai Lunel - Place Guynemer - Quai Rauba Capeu. cyclingstage points
     3091..3141 were replaced by the node geometry of OSM ways 23549762(reversed), 353889269, 1037918098, 4196371, 4196370,
     27627266, 27627267, 362587070 (OpenStreetMap API 0.6 map call, ODbL). Removes 1231 m, inserts 660 m.
  Everything else is the cyclingstage points verbatim (duplicates included).

KNOWN DEFECTS LEFT IN THE DATA ON PURPOSE (no smoothing was applied)
  - 2024, Bd Louis II tunnel: points idx 79-117 (km 0.76-1.15) are flagged tunnel=1; elevation_m there is the ground above
    (about 20 m) instead of the road (about 7 m).
  - 2024, Moyenne Corniche viaduct at Eze: idx 1745-1751 bridge=1; short Corne d'Or tunnel idx 2594-2595.
  - 2023: nine short bridges flagged (longest idx 872-884, on the D13 "Route de Passy", km 11.2-11.3).
  - Both: IGN sampled at points that sit a few metres off the carriageway on steep slopes and hairpins gives point-to-point
    spikes (up to +/-190 %) and inflates raw summed ascent. Use hysteresis or smoothing before computing gradients.
  - Both: 2D track drawn on the road centreline by a route planner, not a recorded ride; start/finish points are
    approximate to a few tens of metres (2024 start is on Route de la Piscine beside Bd Albert Ier).
  - Both tracks contain consecutive duplicate points (70 in 2024, 34 in 2023), i.e. zero-length segments.
