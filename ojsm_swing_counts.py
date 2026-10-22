"""Tracked and competitive swings counted on the same basis, every regular-season pitch (independent check of 2026-10-21).

python3 ojsm_swing_counts.py STATCAST_DIR OUT_JSON

paper_counts.py counts tracked swings over every regular-season pitch (bat speed and swing length both recorded) but competitive
swings from the batter-game panel, which keeps only batter-games with at least 1 PA, so 78 competitive swings in games without a
completed PA were left out of the second count. This counts both over every regular-season pitch with the project's own
functions (common3.adjust, hot_streak_study.pitch_features: Baseball Savant's competitive-swing rule) and reports the panel count
for comparison.
"""
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common3 as C  # noqa: E402
import hot_streak_study as H  # noqa: E402

SC, OUT = sys.argv[1:3]
COLS = ["game_date", "game_pk", "game_type", "game_year", "at_bat_number", "pitch_number", "batter", "pitcher", "stand",
        "p_throws", "home_team", "away_team", "balls", "strikes", "zone", "description", "events", "type",
        "release_speed", "launch_speed", "launch_angle", "launch_speed_angle", "estimated_woba_using_speedangle",
        "woba_value", "woba_denom", "bat_speed", "swing_length"]
files = [f for f in sorted(glob.glob(os.path.join(SC, "*.parquet"))) if any(y in os.path.basename(f) for y in ("2024", "2025", "2026"))]
df = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
df = df[df["game_type"] == "R"].copy()
for c in df.columns:
    dt = str(df[c].dtype)
    if dt in ("string", "str") or dt.startswith("string"):
        df[c] = df[c].astype(object).where(df[c].notna(), None)
for c in ("batter", "pitcher", "game_pk", "game_year"):
    df[c] = df[c].astype("int64")
df["game_date"] = pd.to_datetime(df["game_date"])
df = df.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"]).sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
tracked = int((pd.to_numeric(df["bat_speed"], errors="coerce").notna() & pd.to_numeric(df["swing_length"], errors="coerce").notna()).sum())
d = H.pitch_features(C.adjust(df))
comp = int(d["comp"].sum())
bgp = d.groupby(["batter", "game_pk"])["pa"].sum()
in_pa_games = int(d.set_index(["batter", "game_pk"]).loc[bgp[bgp > 0].index, "comp"].sum())
out = {"note": " ".join(__doc__.split("\n\n")[2].split()), "pitches": int(len(df)), "tracked_swings": tracked,
       "competitive_swings": comp, "competitive_swings_in_batter_games_with_a_pa": in_pa_games}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1))
