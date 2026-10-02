"""Write the 50->50 forecast design matrix (pipeline features) so verify_paper.py can re-estimate the form deciles
with its own estimator.  python3 export_frame.py OUT3_DIR"""
import os
import pickle
import sys

import common3 as C

o3 = sys.argv[1]
bg, lg = pickle.load(open(os.path.join(o3, "_cache", "bg_adj1_comp1.pkl"), "rb"))
W = C.windows(bg, 50, 50, lg)
D = C.prep(W, 200)
cols = ["batter", "game_year", "fwd_woba", "fwd_pa"] + C.STRONG_BASE + C.SWING + ["dev_woba"]
D[cols].to_parquet(os.path.join(o3, "forecast_frame_50_50.parquet"))
print(len(D))
