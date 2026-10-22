"""How much of a real within-season bat-speed decline survives the competitive-swing filter (audit finding, AJSM limitation).

python3 ajsm_filter_attenuation.py OUT_JSON [SIMTEST_JSON]

The filter drops a hitter's swings below his own season 10th percentile (unless exit velocity >= 90 mph). A decline
during part of the season pushes more of that period's swings under the fixed season cutoff, and dropping them raises
the period's mean, so the measured decline is smaller than the true one.
1. Analytic, normal swings, small shift: retention = 1 - lambda(c) * (lambda(c) - c), c = z(0.10), lambda = inverse Mills.
2. Monte Carlo: a season of swings with a decline of d mph in a block of 10% of swings (normal, and a left-skewed
   mixture); the cutoff is the season 10th percentile of all swings, as in the pipeline; the exit-velocity exception is
   ignored (it rescues few slow swings).
3. Optional: the end-to-end check of ajsm_game.py on simulated seasons with injected effects (SIMTEST_JSON).
"""
import json
import sys

import numpy as np
from scipy.stats import norm

out = {}
c = norm.ppf(0.10)
lam = norm.pdf(c) / (1 - norm.cdf(c))
out["analytic_normal_small_shift_retention"] = float(1 - lam * (lam - c))
rng = np.random.default_rng(11)
mc = {}
for shape in ("normal_sd3", "left_skewed_mixture"):
    for d in (0.5, 1.0, 2.0):
        ret = []
        for _ in range(400):
            n = 1500
            if shape == "normal_sd3":
                x = rng.normal(72, 3.0, n)
            else:
                x = np.where(rng.random(n) < 0.9, rng.normal(72.5, 2.5, n), rng.normal(64, 5, n))
            blk = np.zeros(n, bool); blk[:n // 10] = True
            x = x - d * blk
            cut = np.quantile(x, 0.10)
            keep = x >= cut
            meas = x[keep & blk].mean() - x[keep & ~blk].mean()
            ret.append(-meas / d)
        mc[f"{shape}_decline_{d}mph"] = float(np.mean(ret))
out["monte_carlo_retention"] = mc
if len(sys.argv) > 2:
    s = json.load(open(sys.argv[2]))
    w = s["F4a"]["windows"]["oblique"]["1-14"]
    out["pipeline_simulation_check"] = {
        "injected_prodrome_mph": -0.6, "measured_competitive_swings": w["diff"], "measured_all_swings": w["diff_all_swings"],
        "retention_competitive": w["diff"] / -0.6, "retention_all_swings": w["diff_all_swings"] / -0.6,
        "injected_post_return_hand_wrist_mph": -1.0,
        "measured_post_return_competitive": s["F4b"]["effects"]["hand_wrist"]["bat_speed"]["1-10"]["mean"],
        "measured_post_return_all_swings": s["F4b"]["effects"]["hand_wrist"]["bat_speed_all_swings"]["1-10"]["mean"],
        "note": "simulate.py seasons (160 hitters x 3 seasons) with injected IL stints: oblique -0.6 mph in the 14 days before "
                "placement, hand/wrist -1.0 mph in the first 10 games after return; scripts as delivered"}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
