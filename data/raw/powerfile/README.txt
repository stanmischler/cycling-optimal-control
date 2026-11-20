PRO TIME-TRIAL POWER FILE - Michael Hepburn, Tour de France 2018, stage 20 ITT
==============================================================================
Retrieved 2026-11-20. All numbers below are measured values from the rider's own
head-unit file as published by TrainingPeaks; nothing is digitised, estimated or
interpolated.

RACE
----
Race            Tour de France 2018 (UCI WorldTour), stage 20, individual time trial
Date            Saturday 28 July 2018
Course          Saint-Pee-sur-Nivelle -> Espelette (French Basque Country), 31 km,
                rolling/hilly and technical; steep Col de Pinodieta ~3 km from the
                finish (article: "900m climb", ~10 %; analysed segment 810 m).
Rider           Michael Hepburn (AUS), Mitchelton-Scott
Official result 10th, 42:15 (winner Tom Dumoulin 40:52, Hepburn +1:23)
                Source: https://www.cyclingstage.com/tour-de-france-2018-stage-20/stage-20-results-tdf-2018
                (same rank and gap in the TrainingPeaks article below)
Official start  12:34:30 local time, 24th starter, 90 s intervals
                Source: https://cyclingweekly.com/news/racing/tour-de-france/tour-de-france-2018-stage-20-individual-time-trial-start-times-388394

SOURCES
-------
Article   "2018 Tour de France: Stage 20 Power File Analysis", Stephen Gallagher,
          TrainingPeaks blog, published 30 July 2018
          https://www.trainingpeaks.com/blog/2018-tour-de-france-stage-20-power-file-analysis/
File      Public TrainingPeaks activity linked from the article ("Click HERE to view the file")
          https://home.trainingpeaks.com/public/workout/XMAGAHNBMVDBJKJATIU7KWVW3E
          Original device file name recorded in the page: 2018-07-28-11-48-08.fit.gz
          (the .fit itself is not offered for download; only the page data was taken)

HOW IT WAS RETRIEVED
--------------------
One plain HTTP GET of the public activity URL above (curl, generic desktop-browser
user agent, no cookies, no login, no token other than the share key in the URL).
The server returns an HTML page that contains the complete activity inline as a
JavaScript object literal: `var publicActivityWrapper = {...}`. Inside it,
workoutDetailData.workoutSampleList holds channelSet (16 channel names) and
samples (4503 records, each {evt, ms, values[]}). No separate API call was made.
The JSON was parsed with `python3 -I` (json.JSONDecoder.raw_decode); nothing
downloaded was executed. Scripts are outside this folder (session scratchpad).

TERMS-OF-USE CAVEAT (read this)
-------------------------------
The page is deliberately public and no access control was bypassed, but the
TrainingPeaks Terms of Use (https://www.trainingpeaks.com/terms-of-use, "Last
Updated July 28, 2026") say content "may be used only for your personal,
non-commercial, or educational use", that users agree "not to reproduce,
retransmit, distribute, ... publish ... content received or generated through
the Sites", and prohibit "data mining, robots or similar automated means" /
"automated scripts to collect information". The retrieval here was a scripted
fetch of ~25 individual public pages (the blog posts and the public activities
they link, to compare candidates). The same file can be reproduced by opening
the URL in a browser and saving the page. Treat the data as for personal /
educational analysis; do not redistribute the raw page or the trace, and cite
TrainingPeaks / the rider if results are shown.

FILES
-----
raw/trainingpeaks_public_workout_XMAGAHNBMVDBJKJATIU7KWVW3E.html
      The activity page exactly as received (663,840 bytes,
      sha256 e63eca86c645b4245fe2a9934e860bc6a6ec18fab6b7a1c43735d1437eb923e3).
raw/trainingpeaks_blog_2018-tour-de-france-stage-20-power-file-analysis.html
      The article exactly as received.
raw/ST20_01_Hepburn.png, raw/ST20_02_Profile-1024x461.png
      The two figures of the article (TrainingPeaks chart of the ride; stage profile).
raw/embedded_publicActivityWrapper.json
      The JSON object extracted from the page and re-serialised (convenience copy;
      also contains file summary, laps, peak powers, zones). Derived, not "as received".
trace_untrimmed.csv
      All 4503 records of the file: warm-up on the trainer, roll to the start, race,
      roll-out. file_time_s = seconds since recording start (11:48:08 local),
      event = start/stop/none (head-unit timer events; rows with empty values are
      the pause markers present in the source). Extra columns: temperature_c,
      power_left_w, power_right_w, power_balance.
trace.csv
      The race only: 2535 rows, 1 Hz, time_s 0..2534.
      Columns: time_s, power_w, speed_ms, distance_m, elevation_m, lat, lon,
      cadence, heart_rate, temperature_c, file_time_s, file_distance_m.
      distance_m is the file's distance channel re-zeroed at the first race sample
      (file_distance_m keeps the original value). No value was altered otherwise.

HOW THE RACE WAS TRIMMED
------------------------
Start  The head unit was paused (timer 'stop' at file time 2494 s) while the rider
       waited, and resumed ('start' event) at file time 2785 s with the rider
       already at 4.58 m/s and 597 W: that is the start-ramp launch. trace.csv
       starts at that record (time_s = 0 <-> file_time_s = 2785).
       Cross-check with the clock: recording start 11:48:08 + 2785 s = 12:34:33,
       i.e. 3 s after the official start time 12:34:30. So roughly the first 3 s
       of the race (standing start, ~28 m of GPS displacement since the pause
       point, of which part may be the walk onto the ramp) are NOT in the file.
       The file's own "Lap #3" (rider's lap button) also begins here (2786 s).
Finish Last record with pedalling power: file time 5319 s (585 W at 5318 s, 292 W
       at 5319 s, then 0 W while speed decays from 33 to 0 km/h and the unit is
       stopped at 5345 s). trace.csv ends at 5319 s (time_s = 2534). The official
       finish by the clock (12:34:30 + 42:15 = 13:16:45) is file time 5317-5318 s,
       i.e. time_s ~ 2532-2533. Lap #3 in the file ends at 5324.5 s (button pressed
       after the line).
Uncertainty of both ends: about +/-2 s. The exact finish line position is not
marked in the file; if needed, cut at time_s = 2532.

VALIDATION
----------
Duration     trace.csv spans 2534 s (42:14) of recording; + ~3 s missing at the
             start => ~42:17 by the trace vs 42:15 official (within the +/-2 s
             uncertainty of locating the line). By the clock: official start to
             official finish = file time 2782 -> 5317 s.
Sampling     1 Hz, perfectly regular inside the race (all 2534 intervals = 1000 ms,
             no pauses, no gaps).
Missing      0 missing values in every channel over the race.
Zero power   139 samples at exactly 0 W (5.5 %), all in 17 coasting runs of 3-18 s
             at 47-75 km/h (descents / corners) - genuine coasting, not drop-outs.
             177 samples below 50 W.
Power        average 418.4 W, normalised 435.6 W (30 s rolling, 4th power), max 802 W.
             File's own Lap #3: 418 W avg, 436 W NP.
   vs article (Gallagher):
             1st half 21:14: 429 W avg / 439 W NP   -> trace, first 21:14: 430 / 439
             2nd half 20:41: 406 W avg / 431 W NP   -> trace, remainder (21:01): 407 / 431
             peak 5 min 468 W                       -> 468 W (from time_s 863)
             peak 1 min 535 W (foot of the climb)   -> 536 W (from time_s 2160)
             top 1-min speed 69.7 km/h              -> 69.7 km/h
             peak speed 79.7 km/h                   -> 81.7 km/h max 1 s sample
             "5.5 W/kg for 40+ min"                 -> 418.4 W / 77 kg = 5.43 W/kg
             The article's two halves add to 41:55, 20 s less than the official
             time; the author evidently trimmed slightly differently. The article
             gives no whole-ride average.
             (The page-level summary of 323 W avg / 385 W NP is for the whole
             75-min recording including warm-up, not the race.)
Distance     distance channel: 30,741 m over the trimmed race; GPS track
             (haversine sum): 30,769 m. Official course length 31 km, so the trace
             is ~0.8 % short (of which ~10-30 m is the unrecorded start; the rest
             is within normal sensor / official-rounding error).
             Mean speed distance/time = 43.7 km/h; official 31 km / 42:15 = 44.0 km/h.
             NOTE: the speed channel integrates to only 30,315 m (mean 43.05 km/h),
             1.4 % less than the distance channel. Distance and GPS agree with each
             other, so prefer differences of distance_m over speed_ms if it matters.
GPS          Present for every sample, 5 decimal places (~1 m). Start 43.35484 N,
             1.54952 W (Saint-Pee-sur-Nivelle), finish 43.33838 N, 1.44711 W
             (Espelette); max step 22.9 m/s; plausible continuous loop.
Elevation    Present for every sample, 0.01 m resolution (looks barometric),
             28.7 m at the start, 79.3 m at the finish, min 27.4 m, max 185.1 m,
             max step 4.6 m/s (on fast descents). Sum of raw positive steps 769 m,
             negative 718 m (unsmoothed, so somewhat inflated). Not checked
             against a DEM.
Heart rate   avg 158, max 167 bpm. Cadence avg 89 rpm (95 excluding coasting).
Temperature  head-unit sensor 28 C at the start falling to 19-20 C (avg 20.1 C);
             the initial fall is sensor lag, not weather.

RIDER / EQUIPMENT / WEATHER
---------------------------
Body mass   77.0 kg. Not stated explicitly, but implied consistently by every
            W and W/kg pair in the article (429 W = 5.57 W/kg, 495 W = 6.43 W/kg,
            468 W = 6.08 W/kg) and by the file summary (323 W = 4.1948 W/kg).
            This is the weight in his TrainingPeaks account, not a race-day weigh-in.
Threshold   TrainingPeaks account threshold power 410 W (from the page data).
Bike/kit    Not stated in the article or the file. The file contains left/right
            power, torque effectiveness and pedal smoothness, so it is a dual-sided
            power meter; brand not given. Bike, wheels, tyres, CdA, system mass:
            unknown - must be assumed or fitted.
Weather     Article: "rain early in the day"; press previews forecast rain and
            a search-result snippet (source page not opened, so unverified) has
            Hepburn calling the roads "quite slippery".
            He was an early starter (12:34), so damp roads are likely, which also
            explains cautious coasting on descents. No wind data was found; none
            is in the file.

OTHER CAVEATS
-------------
- First ~3 s of the standing start are missing (see above).
- Power is as recorded by the head unit; calibration/zero-offset unknown.
- Hepburn was not a GC rider; he rode flat out for the stage (led for ~2 h).
- Pacing was constrained by a technical, possibly wet course: the 0 W stretches
  are corners/descents, which a pacing optimiser without cornering limits will
  not reproduce.
- Stage 20 of a three-week race: threshold was lower than fresh.

OTHER CANDIDATES FOUND (same mechanism, not processed)
------------------------------------------------------
All are public TrainingPeaks activities linked from TrainingPeaks blog posts;
each page embeds 1 Hz samples in the same way. Summaries are for the whole file.
- Roman Kreuziger, Tour 2017 stage 20 ITT Marseille (22.5 km, one sharp climb):
  22.9 km, 32.9 min, 341 W / NP 371, HR + GPS + elevation.
  https://home.trainingpeaks.com/public/workout/MLY22RVN5YNZRC2IZJU3SVYNU4
  article https://www.trainingpeaks.com/blog/stage-19-20-power-analysis-rigoberto-urans-road-glory/
  (the Uran link in that article is his stage 19 road stage, not the TT)
- Oscar Gatto, Tour 2016 stage 13 ITT Bourg-Saint-Andeol - La Caverne du
  Pont-d'Arc (37.5 km, hilly, windy): 36.5 km, 56.6 min, 332 W / NP 343, GPS +
  elevation, no HR; not a full-gas ride by a specialist.
  https://home.trainingpeaks.com/public/workout/2YZL642ENYAID3IPSJRM6FGLSI
  article https://www.trainingpeaks.com/blog/power-analysis-2016-tour-de-france-week-2/
- Michael Valgren, Tour 2016 stage 18 uphill ITT Sallanches - Megeve (17 km):
  19.3 km file incl. extra, 276 W / NP 335, GPS + elevation, no HR.
  https://home.trainingpeaks.com/public/workout/KTZ7RK75ELJQBDYGGNSC332JUY
  article https://www.trainingpeaks.com/blog/power-analysis-2016-tour-de-france-week-3/
- Michael Morkov, Tour 2014 stage 20 ITT Bergerac - Perigueux (54 km): 55.1 km,
  282 W / NP 319, HR, but NO GPS.
  https://home.trainingpeaks.com/public/workout/DT5SBW6CCMLRTKXHBKIK7P4T4I
  article https://www.trainingpeaks.com/blog/2014-tour-de-france-week-3-power-analysis/
- Michael Morkov, Tour of California 2015 stage 6 ITT: tpks.ws short links in
  https://www.trainingpeaks.com/blog/power-analysis-2015-tour-of-california/
  (links not followed, so unverified).
Hepburn's file was chosen because it is complete (power, speed, distance,
elevation, GPS, cadence, HR, all at 1 Hz with no gaps), the course is hilly, the
rider is a TT specialist who rode it flat out to a top-10, and the article quotes
enough numbers to cross-check the trace.
