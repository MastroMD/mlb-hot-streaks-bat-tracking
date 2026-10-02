"""
Download regular-season Statcast pitch data for 2024-2026 (the bat-tracking era)
and save a slim copy for the hot-streak study.

Run on your Mac:   python3 fetch_statcast.py
Output:            ./statcast_hotstreak/statcast_YYYY_MM.parquet  (one file per month)

It is resumable: months already saved are skipped, so if Savant times out
just run it again.
"""
import os
import sys
import time
from datetime import date, timedelta

import pandas as pd

try:
    from pybaseball import statcast, cache
except ImportError:
    sys.exit("pybaseball is not installed. Run:  pip3 install pybaseball pyarrow")

cache.enable()

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "statcast_hotstreak")
os.makedirs(OUT, exist_ok=True)

SEASONS = {  # generous windows; spring/postseason rows are filtered out below
    2024: (date(2024, 3, 15), date(2024, 10, 1)),
    2025: (date(2025, 3, 15), date(2025, 9, 30)),
    2026: (date(2026, 3, 20), min(date(2026, 9, 29), date.today() - timedelta(days=1))),
}

KEEP = [
    "game_date", "game_pk", "game_type", "game_year", "at_bat_number", "pitch_number",
    "batter", "pitcher", "stand", "p_throws", "home_team", "away_team", "inning_topbot",
    "balls", "strikes", "zone", "plate_x", "plate_z", "sz_top", "sz_bot",
    "pitch_type", "release_speed", "description", "events", "type", "bb_type",
    "launch_speed", "launch_angle", "launch_speed_angle",
    "estimated_woba_using_speedangle", "woba_value", "woba_denom",
    "bat_speed", "swing_length", "attack_angle", "swing_path_tilt",
    "delta_run_exp",
    # added 2026-09-30 for the pitcher analog (A5): not in the extract used for the SSAC27 abstract
    "release_spin_rate", "release_extension", "release_pos_z",
]


def month_chunks(start, end):
    cur = start
    while cur <= end:
        nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
        yield cur, min(end, nxt - timedelta(days=1))
        cur = nxt


def save(df, path_base):
    try:
        df.to_parquet(path_base + ".parquet", index=False)
    except Exception:
        df.to_csv(path_base + ".csv.gz", index=False)


for yr, (s, e) in SEASONS.items():
    for a, b in month_chunks(s, e):
        base = os.path.join(OUT, f"statcast_{a:%Y_%m}")
        if os.path.exists(base + ".parquet") or os.path.exists(base + ".csv.gz"):
            print(f"skip {a:%Y-%m} (already saved)")
            continue
        for attempt in range(3):
            try:
                print(f"downloading {a} -> {b} ...", flush=True)
                df = statcast(start_dt=a.isoformat(), end_dt=b.isoformat(), verbose=False)
                break
            except Exception as ex:  # Savant is flaky; back off and retry
                print(f"  retry {attempt + 1}: {ex}")
                time.sleep(20)
        else:
            print(f"  FAILED {a:%Y-%m}; re-run the script later to fill it in")
            continue
        if df is None or df.empty:
            print("  no games")
            continue
        df = df[[c for c in KEEP if c in df.columns]]
        df = df[df["game_type"] == "R"]
        save(df, base)
        print(f"  saved {len(df):,} pitches")

files = sorted(os.listdir(OUT))
print(f"\nDone. {len(files)} files in {OUT}")
print("Next: point hot_streak_study.py at that folder (see README, Reproduce).")
