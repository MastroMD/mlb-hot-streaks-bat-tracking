"""Monte Carlo spread of the shuffled-null slope (50 -> 50 PA, park/week adjusted): 20 independent shuffles.
python3 null_mc.py OUT_DIR   (reads OUT_DIR/_cache/bg_adj1_comp1.pkl written by analysis3.py)"""
import json
import os
import pickle
import sys

import numpy as np

import common3 as C

out = sys.argv[1]
bg, lg = pickle.load(open(os.path.join(out, "_cache", "bg_adj1_comp1.pkl"), "rb"))
M = ["swing_len", "bat_speed", "fast_swing", "z_swing", "chase", "whiff", "barrel", "k_rate", "ev", "xwoba", "hard_hit", "woba"]
W = C.windows(bg, 50, 50, lg)
act = {m: C.slope(W, m) for m in M}
nulls = {m: [] for m in M}
for s in range(1000, 1020):
    W0 = C.windows(C.shuffle_games(bg, s), 50, 50, lg)
    for m in M:
        nulls[m].append(C.slope(W0, m)[0])
res = {}
for m in M:
    a, ase, n = act[m]
    nl = np.array(nulls[m])
    res[m] = {"slope_actual": a, "se_actual": ase, "null_mean": float(nl.mean()), "null_sd_across_shuffles": float(nl.std(ddof=1)),
              "persistence_mean_null": float(a - nl.mean()), "persistence_range": [float(a - nl.max()), float(a - nl.min())]}
    print(m, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[m].items()})
json.dump(res, open(os.path.join(out, "null_mc.json"), "w"), indent=1)
