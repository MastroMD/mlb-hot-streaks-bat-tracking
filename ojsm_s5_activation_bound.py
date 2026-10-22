"""How many observations of the original injury analysis could the added activations move? (author items of 2026-10-21,
PLAN_ADDENDUM_2026-10-17h.md item 14; post hoc, descriptive)

python3 ojsm_s5_activation_bound.py MISSING_ACTIVATIONS_JSON ORIGINAL_IL_FILE STATCAST_DIR PAPER_RESULTS_JSON OUT_JSON

The IL rows of Supplemental Table S5 come from the original injury analysis (analysis3.py A8, on ORIGINAL_IL_FILE,
repo/data_public/il_transactions_hitters_2024_2026.txt), which drops from its first row every game within 45 days after an
activation. It was not rerun with the activations added in the final run (MISSING_ACTIVATIONS_JSON, ojsm_missing_activations.py).
This counts the regular-season games of each such hitter from the added activation through 45 days later that are not already
within 45 days after an activation of ORIGINAL_IL_FILE: an upper bound on the observations of that analysis (one per game with a
complete window, PAPER_RESULTS_JSON['A8']['n_points']) that adding the activations could move out of its first row.
"""
import glob
import json
import os
import sys

import pandas as pd

MAJ, ILO, SC, PR, OUT = sys.argv[1:6]
MA = json.load(open(MAJ))
files = [f for f in sorted(glob.glob(os.path.join(SC, "*.parquet"))) if any(y in os.path.basename(f) for y in ("2024", "2025", "2026"))]
g = pd.concat([pd.read_parquet(f, columns=["game_date", "game_type", "batter", "game_pk"]) for f in files], ignore_index=True)
g = g[g["game_type"] == "R"].drop_duplicates(["batter", "game_pk"])
g["game_date"] = pd.to_datetime(g["game_date"])
act = {}
for line in open(ILO):
    line = line.strip()
    if line:
        b, ev = line.split(":")
        act[int(b)] = [pd.Timestamp("20" + e[:6]) for e in ev.split() if e[6] == "A"]
rows, tot = [], 0
for r in MA["records"]:
    b, a = int(r["batter"]), pd.Timestamp(r["first_game_after"])
    days = g.loc[(g["batter"] == b) & (g["game_date"] >= a) & (g["game_date"] <= a + pd.Timedelta(days=45)), "game_date"]
    new = [d for d in days if not any(x <= d <= x + pd.Timedelta(days=45) for x in act.get(b, []))]
    rows.append({"batter": b, "placement": r["placement"], "games_within_45_days": int(len(days)), "not_already_after_an_activation": len(new)})
    tot += len(new)
n_points = json.load(open(PR))["A8"]["n_points"]
out = {"note": " ".join(__doc__.split("\n\n")[2].split()), "games_bound": tot, "a8_n_points": n_points, "share": tot / n_points, "by_placement": rows}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("games_bound", "a8_n_points", "share")}))
