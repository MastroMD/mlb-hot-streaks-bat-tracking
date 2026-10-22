"""Addendum H5 follow-up (post hoc, descriptive): the hand/wrist stints followed to games 41-60, their deficit in earlier
windows, and their mechanism; same for the stints followed to games 21-40. Reads the stint-level rows written by
ajsm_game_ojsm.py. python3 ojsm_new_analyses_b.py RESULTS_DIR CODED_CSV"""
import json, os, sys
import numpy as np, pandas as pd, statsmodels.api as sm
RES, CODED = sys.argv[1], sys.argv[2]
B = pd.read_csv(os.path.join(RES, "ojsm_f4b_stint_rows.csv"))
ST = pd.read_csv(os.path.join(RES, "ojsm_f4b_stints.csv"), parse_dates=["S", "R"])
CD = pd.read_csv(CODED, parse_dates=["day"]).drop_duplicates(["batter", "day"])
ST = ST.merge(CD[["batter", "day", "mechanism"]], left_on=["batter", "S"], right_on=["batter", "day"], how="left")
def cm(y, g):
    f = sm.OLS(np.asarray(y, float), np.ones(len(y))).fit(cov_type="cluster", cov_kwds={"groups": np.asarray(g)})
    return {"est": float(f.params[0]), "lo": float(f.conf_int()[0][0]), "hi": float(f.conf_int()[0][1]), "n": int(len(y))}
out = {}
for col in ("diff", "diff_all"):
    x = B[(B["met"] == "bat_speed") & (B["fam"] == "hand_wrist")].pivot_table(index=["stint", "batter"], columns="win", values=col).reset_index()
    r = {}
    for reach in ("21-40", "41-60"):
        s = x[x[reach].notna()]
        r[f"reached_{reach}"] = {w: cm(s[w].dropna(), s.loc[s[w].notna(), "batter"]) for w in ("1-10", "21-40", "41-60") if s[w].notna().sum() >= 5}
        mech = ST.set_index("stint").loc[s["stint"], "mechanism"].value_counts().to_dict()
        r[f"reached_{reach}"]["mechanisms"] = {k: int(v) for k, v in mech.items()}
        r[f"reached_{reach}"]["n_fracture_or_surgery"] = int(sum(v for k, v in mech.items() if k in ("fracture", "surgery")))
        r[f"reached_{reach}"]["n"] = int(len(s))
    s = x[x["41-60"].isna()]
    r["not_reached_41-60"] = {w: cm(s[w].dropna(), s.loc[s[w].notna(), "batter"]) for w in ("1-10", "21-40") if s[w].notna().sum() >= 5}
    # 2026-10-17 audit: the difference between the 2 groups in games 1 to 10 (hitter-clustered), and equivalence in games 21 to 40
    # of the stints followed to games 41 to 60 (TOST against +/-0.5 mph), so that the text compares like with like
    g = x[x["1-10"].notna()].assign(reach=lambda d: d["41-60"].notna().astype(float))
    f2 = sm.OLS(g["1-10"].to_numpy(float), sm.add_constant(g[["reach"]]).to_numpy(float)).fit(cov_type="cluster", cov_kwds={"groups": g["batter"].to_numpy()})
    r["reached_minus_not_1-10"] = {"est": float(f2.params[1]), "lo": float(f2.conf_int()[1][0]), "hi": float(f2.conf_int()[1][1]), "p": float(f2.pvalues[1]),
                                    "n_reached": int(g["reach"].sum()), "n_not": int((1 - g["reach"]).sum())}
    s = x[x["41-60"].notna() & x["21-40"].notna()]
    f3 = sm.OLS(s["21-40"].to_numpy(float), np.ones(len(s))).fit(cov_type="cluster", cov_kwds={"groups": s["batter"].to_numpy()})
    e, se = float(f3.params[0]), float(f3.bse[0]); z90 = 1.6448536269514722
    from scipy.stats import norm
    r["reached_41-60_tost_21-40"] = {"est": e, "se": se, "lo90": e - z90 * se, "hi90": e + z90 * se, "n": int(len(s)),
                                      "tost_p": float(max(norm.sf((e + 0.5) / se), norm.sf((0.5 - e) / se))), "equivalent": bool(e - z90 * se > -0.5 and e + z90 * se < 0.5)}
    out[col] = r
json.dump(out, open(os.path.join(RES, "ojsm_new_analyses_b.json"), "w"), indent=1)
print(json.dumps(out["diff"], indent=1))
