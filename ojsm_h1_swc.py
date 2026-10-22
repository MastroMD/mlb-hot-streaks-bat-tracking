"""Plan addendum H, item 1 (post hoc): smallest worthwhile change in bat speed = 0.2 x the between-hitter SD of season-mean
bat speed (Hopkins et al 2009), from FanGraphs season lines of hitters with >= 200 competitive swings, 2024-2026 (one SD per
season, then the mean of the 3 SDs). The 100-swing SDs are a sensitivity check.

python3 ojsm_h1_swc.py "inputs_fangraphs_not_redistributed/Fangraphs Bat Tracking ALL.csv" repo/results/ojsm_h1_swc.json"""
import json
import sys

import pandas as pd

src, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(src, encoding="utf-8-sig")
d = d[d["Season"].between(2024, 2026)]
res = {}
for y, g in d[d["CompSw"] >= 200].groupby("Season"):
    res[str(y)] = {"n": int(len(g)), "mean": float(g["BatSpd"].mean()), "sd": float(g["BatSpd"].std(ddof=1))}
res["mean_of_season_sds"] = sum(res[str(y)]["sd"] for y in (2024, 2025, 2026)) / 3
res["swc_0p2sd"] = 0.2 * res["mean_of_season_sds"]
res["sd_100plus_by_season"] = {str(y): float(g["BatSpd"].std(ddof=1)) for y, g in d[d["CompSw"] >= 100].groupby("Season")}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res))
