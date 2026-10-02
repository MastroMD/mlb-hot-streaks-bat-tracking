"""Calendar length of the A1 forward windows (descriptive; added 2026-10-01 to express PA horizons in weeks).
python3 horizon_weeks.py OUT3_DIR   (reads OUT3_DIR/_cache/bg_adj1_comp1.pkl) -> OUT3_DIR/horizon_weeks.json
For every A1 fixed-sample decision point (50 PA behind, 200 PA ahead within the season, hitter-season >= 300 PA), the
calendar span of the next 25/50/100/150/200 PA = days from the decision game to the game that completes the window."""
import json
import os
import pickle
import sys

import numpy as np

import common3 as C

o3 = sys.argv[1]
bg, lg = pickle.load(open(os.path.join(o3, "_cache", "bg_adj1_comp1.pkl"), "rb"))
bg = bg.sort_values(["batter", "game_date", "game_pk"]).reset_index(drop=True)
W = C.windows(bg, 50, 200, lg)
W = W[(W["season_pa"] >= 300) & (W["rec_bat_speed__n"] >= 30) & (W["fwd_bat_speed__n"] >= 30)]
pa = bg["pa"].to_numpy(float)
day = bg["game_date"].to_numpy().astype("datetime64[D]").astype(np.int64)
cpa = np.concatenate([[0], np.cumsum(pa)])
out = {"n_decision_points": int(len(W))}
for h in (25, 50, 100, 150, 200):
    i = W["row"].to_numpy()
    k = np.searchsorted(cpa, cpa[i] + h, side="left") - 1      # last game of the window
    span = day[k] - day[i] + 1
    out[str(h)] = {"median_days": float(np.median(span)), "iqr_days": [float(np.percentile(span, 25)), float(np.percentile(span, 75))],
                   "median_weeks": float(np.median(span) / 7)}
json.dump(out, open(os.path.join(o3, "horizon_weeks.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
