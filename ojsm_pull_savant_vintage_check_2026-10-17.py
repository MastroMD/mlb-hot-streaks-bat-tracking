"""Re-pull regular-season Statcast pitch data 2024-2026 from Baseball Savant, one request per date.
Mirrors repo/fetch_statcast.py KEEP columns (plus attack_direction and a few context fields).
Resumable: dates already saved are skipped. Writes daily CSVs then per-season parquet."""
import os, sys, time, io, csv, concurrent.futures as cf
from datetime import date, timedelta
import urllib.request, urllib.error
import pandas as pd

OUT = "/home/claude/work/statcast_pull/daily"
SEASONS = {2024: (date(2024, 3, 18), date(2024, 10, 1)),
           2025: (date(2025, 3, 16), date(2025, 9, 30)),
           2026: (date(2026, 3, 23), date(2026, 9, 29))}
KEEP = ["game_date", "game_pk", "game_type", "game_year", "at_bat_number", "pitch_number",
        "batter", "pitcher", "stand", "p_throws", "home_team", "away_team", "inning", "inning_topbot", "outs_when_up",
        "balls", "strikes", "zone", "plate_x", "plate_z", "sz_top", "sz_bot",
        "pitch_type", "release_speed", "effective_speed", "description", "events", "type", "bb_type",
        "launch_speed", "launch_angle", "launch_speed_angle",
        "estimated_woba_using_speedangle", "woba_value", "woba_denom",
        "bat_speed", "swing_length", "attack_angle", "attack_direction", "swing_path_tilt",
        "delta_run_exp", "release_spin_rate", "release_extension", "release_pos_z"]
URL = ("https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfGT=R%7C&player_type=batter"
       "&game_date_gt={d}&game_date_lt={d}&type=details")

def fetch(d):
    path = os.path.join(OUT, f"{d.isoformat()}.csv")
    if os.path.exists(path):
        return d, "skip", 0
    last = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(URL.format(d=d.isoformat()), headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                raw = r.read().decode("utf-8-sig")
            df = pd.read_csv(io.StringIO(raw), low_memory=False)
            if len(df) == 0 or "game_date" not in df.columns:
                pd.DataFrame(columns=KEEP).to_csv(path, index=False)
                return d, "empty", 0
            df = df[[c for c in KEEP if c in df.columns]]
            df.to_csv(path + ".tmp", index=False)
            os.replace(path + ".tmp", path)
            return d, "ok", len(df)
        except Exception as ex:
            last = ex
            time.sleep(10 * (attempt + 1))
    return d, f"FAIL {last}", 0

def dates():
    for yr, (s, e) in SEASONS.items():
        cur = s
        while cur <= e:
            yield cur
            cur += timedelta(days=1)

if __name__ == "__main__":
    nthreads = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    ds = list(dates())
    t0 = time.time(); tot = 0
    with cf.ThreadPoolExecutor(nthreads) as ex:
        for i, (d, st, n) in enumerate(ex.map(fetch, ds)):
            tot += n
            if st.startswith("FAIL") or i % 20 == 0:
                print(f"{i+1}/{len(ds)} {d} {st} {n} total={tot:,} {time.time()-t0:.0f}s", flush=True)
    print("DONE", tot, f"{time.time()-t0:.0f}s", flush=True)
