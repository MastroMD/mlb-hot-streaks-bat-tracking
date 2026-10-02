"""Exploratory checks requested by Reviewer 2 (plan addendum E, PLAN_ADDENDUM_2026-10-02e.md). Labelled exploratory.

python3 e_review.py DATA_DIR PUBLIC_DIR OUT3_DIR   -> OUT3_DIR/e_review.json
E2  Green and Zwiebel's hot and cold definition (A9 code path) with the null averaged over 20 shuffles.
E3  Lineup decisions: realised check for rule (b), and concentration of decisions by team-season and by hitter.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

data, pub, out3 = sys.argv[1:4]
sys.argv = ["analysis3.py", data, pub, out3, "NONE"]
import analysis3 as A  # noqa: E402

C = A.C
out = {"plan": "PLAN_ADDENDUM_2026-10-02e.md", "label": "exploratory"}

# ------------------------------------------------------------------ E2
bg, lg = A.panel()
DEFS = [("woba", "rec_woba", "orc_woba_50"), ("onb", "rec_onb", "orc_onb_50"), ("hr", "rec_hr", "orc_hr_50"),
        ("bat_speed", "rec_bat_speed", "orc_bat_speed_50")]
OUTS = [("woba", "fwd_woba", "orc_woba_50", 1000), ("onb", "fwd_onb", "orc_onb_50", 1000), ("hr", "fwd_hr", "orc_hr_50", 100)]


def gz(b):
    W25 = C.windows(b, 25, 25, lg, extra=("onb", "hr"))
    W50 = C.windows(b, 50, 50, lg, extra=("onb", "hr"))
    J = W25.merge(W50[["row"] + [c for c in W50.columns if c.startswith("orc_")]], on="row", suffixes=("", "_50"))
    J = J[J["season_pa"] >= 300]
    res = {}
    for hot_m, rc, oc0 in DEFS:
        Jm = J.dropna(subset=[rc, oc0])
        if hot_m == "bat_speed":
            Jm = Jm[(Jm["rec_bat_speed__n"] >= 20) & (Jm["orc_bat_speed__n_50"] >= 60)]
        dev = Jm[rc] - Jm[oc0]
        hot, cold = (dev >= dev.quantile(0.95)).astype(float), (dev <= dev.quantile(0.05)).astype(float)
        res[hot_m] = {}
        for outc, fc, oc, scale in OUTS:
            y = (Jm[fc] - Jm[oc]) * scale
            f = C.ols_cl(y.to_numpy(), np.column_stack([hot, cold]), Jm["batter"].to_numpy())
            res[hot_m][outc] = {"hot": float(f.params[1]), "hot_se": float(f.bse[1]), "cold": float(f.params[2]), "cold_se": float(f.bse[2]),
                                "base_rate": float(Jm[oc].mean() * scale)}
    return res


act = gz(bg)
nulls = [gz(C.shuffle_games(bg, s)) for s in range(2000, 2020)]
E2 = {}
for d in act:
    E2[d] = {}
    for o in act[d]:
        a = act[d][o]
        nh = np.array([n[d][o]["hot"] for n in nulls]); nc = np.array([n[d][o]["cold"] for n in nulls])
        sh = float(np.mean([n[d][o]["hot_se"] for n in nulls])); sc = float(np.mean([n[d][o]["cold_se"] for n in nulls]))
        E2[d][o] = {"hot_raw": a["hot"], "cold_raw": a["cold"], "null_hot_mean": float(nh.mean()), "null_cold_mean": float(nc.mean()),
                    "null_hot_sd": float(nh.std(ddof=1)), "null_cold_sd": float(nc.std(ddof=1)),
                    "hot": a["hot"] - float(nh.mean()), "hot_se": float(np.hypot(a["hot_se"], sh)),
                    "cold": a["cold"] - float(nc.mean()), "cold_se": float(np.hypot(a["cold_se"], sc)), "base_rate": a["base_rate"]}
out["E2_green_zwiebel_20_shuffles"] = E2
print("E2", json.dumps(E2)[:800], flush=True)

# ------------------------------------------------------------------ E3
E3 = {}
for rule, key in (("plan", "prespecified"), ("mutual", "refined_exploratory")):
    A.A7_core(rule)
    Dd = A.A7_ROWS[rule]
    ch = Dd[Dd["b"] != Dd["a"]]
    real = np.where(ch["b"] == "L", ch["resid_L"] - ch["resid_R"], ch["resid_R"] - ch["resid_L"])
    g = ch[["team", "yr", "L", "R"]].astype(str).agg("_".join, axis=1).to_numpy()
    m = sm.OLS(real, np.ones(len(real))).fit(cov_type="cluster", cov_kwds={"groups": g})
    ts = Dd.groupby(["team", "yr"]).size().sort_values(ascending=False)
    per_hitter = pd.concat([Dd[["team", "yr", "L", "R"]].drop_duplicates()["L"], Dd[["team", "yr", "L", "R"]].drop_duplicates()["R"]]).value_counts()
    E3[key] = {"b_changed_n": int(len(ch)), "b_realised_minus_platoon_pick_pts": [float(m.params[0]), float(m.bse[0])],
               "largest_team_season": [str(ts.index[0][0]), int(ts.index[0][1]), int(ts.iloc[0])],
               "largest_team_season_share": float(ts.iloc[0] / len(Dd)), "max_pairs_per_hitter": int(per_hitter.max()),
               "n_decisions": int(len(Dd))}
out["E3_lineups"] = E3
print("E3", json.dumps(E3), flush=True)
json.dump(out, open(os.path.join(out3, "e_review.json"), "w"), indent=1)
