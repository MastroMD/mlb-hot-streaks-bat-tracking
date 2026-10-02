"""Descriptive sample counts quoted in the paper's Data and Methods sections (no estimation).
python3 paper_counts.py DATA_DIR OUT3_DIR   (OUT3_DIR holds the analysis3.py panel cache) -> OUT3_DIR/paper_counts.json"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

import common3 as C

data, o3 = sys.argv[1:3]
bg, lg = pickle.load(open(os.path.join(o3, "_cache", "bg_adj1_comp1.pkl"), "rb"))
d = pd.concat([pd.read_parquet(os.path.join(data, f), columns=["game_type", "game_year", "game_date", "game_pk", "batter", "pitcher", "woba_denom",
                                                                "bat_speed", "swing_length", "description"])
               for f in sorted(os.listdir(data)) if f.endswith(".parquet")])
d = d[d["game_type"] == "R"]
out = {"n_pa": int((pd.to_numeric(d["woba_denom"], errors="coerce") == 1).sum()),
       "n_games": int(d["game_pk"].nunique()),
       "n_games_by_season": {int(y): int(g["game_pk"].nunique()) for y, g in d.groupby("game_year")},
       "n_pitchers": int(d["pitcher"].nunique()),
       "n_tracked_swings": int((pd.to_numeric(d["bat_speed"], errors="coerce").notna() & pd.to_numeric(d["swing_length"], errors="coerce").notna()).sum()),
       "n_competitive_swings": int(bg["comp"].sum()),
       "n_batter_games": int(len(bg))}
bsd = d[pd.to_numeric(d["bat_speed"], errors="coerce").notna()]
out["first_tracked_date"] = {int(y): str(pd.to_datetime(g["game_date"]).min().date()) for y, g in bsd.groupby("game_year")} if "game_date" in bsd else None
out["first_game_date"] = {int(y): str(pd.to_datetime(g["game_date"]).min().date()) for y, g in d.groupby("game_year")} if "game_date" in d else None
hs = bg.groupby(["batter", "game_year"])["pa"].sum()
out["n_hitter_seasons_300pa"] = int((hs >= 300).sum())
out["n_hitters_300pa"] = int(hs[hs >= 300].reset_index()["batter"].nunique())
W = C.windows(bg, 50, 50, lg)
mn = C.H.MIN_N["comp"]
ok = (W["season_pa"] >= 300) & (W["rec_bat_speed__n"] >= mn) & (W["fwd_bat_speed__n"] >= mn) & (W["orc_bat_speed__n"] >= 3 * mn)
S = W[ok]
out["persistence_sample_bat_speed"] = {"n_points": int(len(S)), "n_hitters": int(S["batter"].nunique()),
                                       "n_hitter_seasons": int(S[["batter", "game_year"]].drop_duplicates().shape[0])}
out["min_n"] = C.H.MIN_N
json.dump(out, open(os.path.join(o3, "paper_counts.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
