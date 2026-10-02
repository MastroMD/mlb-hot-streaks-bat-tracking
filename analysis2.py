"""Robustness pass:
 1. adjust bat speed / swing length / exit velocity for park (Hawk-Eye calibration, climate) and
    league-wide week-of-season drift before measuring anyone's form
 2. a stronger baseline forecast (full history of every metric + season-to-date), so "form" only gets
    credit for what a good projection would miss
 3. leave-one-season-out cross-fitting for the composite form index (all three seasons out of sample)
 4. cluster-robust top/bottom-decile effects
python3 analysis2.py DATA_DIR OUT_DIR
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

import hot_streak_study as H

data_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

d = H.load(data_dir)
wk = d["game_date"].dt.isocalendar().week.astype(int)
for c in ["bat_speed", "swing_length", "launch_speed"]:
    x = pd.to_numeric(d[c], errors="coerce")
    ymean = x.groupby(d["game_year"]).transform("mean")
    park = x.groupby([d["home_team"], d["game_year"]]).transform("mean") - ymean
    week = x.groupby([d["game_year"], wk]).transform("mean") - ymean
    d[c] = x - park - week
    print(f"{c}: park effect sd {park.groupby([d['home_team'], d['game_year']]).first().std():.2f}, "
          f"week drift range {week.min():.2f}..{week.max():.2f}")
d = H.pitch_features(d)
d = H.context(d)
bg = H.batter_games(d)
lg = H.league_means(bg)
PLATOON = H.platoon(d)

ALLM = list(H.METRICS)
STRONG_BASE = ([f"prior_{m}" for m in ALLM] + ["std_woba", "std_xwoba", "std_bat_speed", "std_ev", "std_k_rate",
               "std_chase", "std_whiff"] + ["fwd_oppq", "fwd_park", "fwd_same"])


def prep(W):
    D = W[(W["prior_pa"] >= 200)].copy()
    for m in ALLM:
        _, dn, kreg, _ = H.METRICS[m]
        n = D[f"rec_{m}__n"]
        D[f"dev_{m}"] = ((D[f"rec_{m}"] - D[f"prior_{m}"]) * n / (n + kreg)).fillna(0.0)
        D[f"raw_{m}"] = D[f"rec_{m}"] - D[f"prior_{m}"]
    for c in STRONG_BASE:
        D[c] = D[c].fillna(D[c].mean())
    return D


def resid_strong(D, target):
    y = D[f"fwd_{target}"] * 1000
    f = sm.WLS(y, sm.add_constant(D[STRONG_BASE]), weights=D["fwd_pa"]).fit()
    return y - f.predict(sm.add_constant(D[STRONG_BASE])), f


def decile_effects(D, resid, var):
    ok = D[var].notna()
    D, resid = D[ok], resid[ok]
    q = pd.qcut(D[var].rank(method="first"), 10, labels=False)
    # reference group = the two middle deciles (a "typical" recent stretch)
    X = pd.get_dummies(q, prefix="d", dtype=float).drop(columns=["d_4", "d_5"])
    fit = sm.WLS(resid, sm.add_constant(X), weights=D["fwd_pa"]).fit(cov_type="cluster", cov_kwds={"groups": D["batter"]})
    means = D.groupby(q)[var].mean()
    out = {"decile": list(range(10)), "x_mean": [float(v) for v in means],
           "bump": [], "se": []}
    for k in range(10):
        if k in (4, 5):
            out["bump"].append(0.0); out["se"].append(0.0)
        else:
            out["bump"].append(float(fit.params[f"d_{k}"])); out["se"].append(float(fit.bse[f"d_{k}"]))
    return out


def crossfit_form(D, resid, cols):
    """Leave-one-season-out: form index fitted on the other seasons, evaluated on the held-out one."""
    comp = pd.Series(np.nan, index=D.index)
    for yr in sorted(D["game_year"].unique()):
        tr, te = D["game_year"] != yr, D["game_year"] == yr
        f = sm.WLS(resid[tr], sm.add_constant(D.loc[tr, cols]), weights=D.loc[tr, "fwd_pa"]).fit()
        comp[te] = f.predict(sm.add_constant(D.loc[te, cols], has_constant="add")) - f.params["const"]
    D = D.assign(form=comp)
    eff = decile_effects(D, resid, "form")
    # calibration slope: realised vs predicted
    cal = sm.WLS(resid, sm.add_constant(comp), weights=D["fwd_pa"]).fit(cov_type="cluster", cov_kwds={"groups": D["batter"]})
    full = sm.WLS(resid, sm.add_constant(D[cols]), weights=D["fwd_pa"]).fit(cov_type="cluster", cov_kwds={"groups": D["batter"]})
    mse0 = np.average(resid ** 2, weights=D["fwd_pa"])
    mse1 = np.average((resid - comp) ** 2, weights=D["fwd_pa"])
    return {"deciles": eff, "calibration_slope": [float(cal.params.iloc[1]), float(cal.bse.iloc[1])],
            "oos_r2_gain": float(1 - mse1 / mse0),
            "coefs": {c: [float(full.params[c]), float(full.bse[c])] for c in cols}}, comp


results = {"platoon": PLATOON, "league": lg}
for rp, fp in [(50, 50), (50, 25), (25, 25), (100, 50)]:
    tag = f"rec{rp}_fwd{fp}"
    print("==", tag, flush=True)
    W = H.build_windows(bg, rp, fp, lg)
    W0 = H.build_windows(bg, rp, fp, lg, shuffle=True)
    pers = H.frame_a(W, W0)
    cross = H.cross_predict(W, W0, "woba")
    print(pers[["metric", "persistence", "se"]].round(3).to_string(index=False))
    D = prep(W)
    R = {"persistence": pers.to_dict(orient="records"), "cross_woba": cross.to_dict(orient="records"),
         "n_points": int(len(D)), "n_hitters": int(D["batter"].nunique())}
    for target in ["woba", "xwoba"]:
        resid, base = resid_strong(D, target)
        single = {m: decile_effects(D, resid, f"raw_{m}") for m in ["woba", "xwoba", "bat_speed", "swing_len",
                  "fast_swing", "ev", "hard_hit", "barrel", "blast", "squared_up", "chase", "z_swing", "whiff", "k_rate"]}
        forms = {}
        for name, cols in [("results_only", ["dev_woba"]),
                           ("swing_only", ["dev_bat_speed", "dev_fast_swing", "dev_swing_len"]),
                           ("process", [f"dev_{m}" for m in H.PROCESS]),
                           ("everything", [f"dev_{m}" for m in ALLM])]:
            forms[name], comp = crossfit_form(D, resid, cols)
            print(f"  {target} {name:12s} r2gain {forms[name]['oos_r2_gain']:.4f} cal {forms[name]['calibration_slope'][0]:.2f} "
                  f"top {forms[name]['deciles']['bump'][9]:.1f}±{forms[name]['deciles']['se'][9]:.1f} "
                  f"bot {forms[name]['deciles']['bump'][0]:.1f}±{forms[name]['deciles']['se'][0]:.1f}")
        for m in ["woba", "bat_speed", "ev", "xwoba"]:
            e = single[m]
            print(f"  {target} single {m:10s} top {e['bump'][9]:.1f}±{e['se'][9]:.1f}  bot {e['bump'][0]:.1f}±{e['se'][0]:.1f}"
                  f"   (x: {e['x_mean'][0]:.3f} .. {e['x_mean'][9]:.3f})")
        R[target] = {"single": single, "forms": forms}
    results[tag] = R

with open(os.path.join(out_dir, "results2.json"), "w") as f:
    json.dump(results, f, default=lambda o: o.item() if hasattr(o, "item") else str(o), indent=1)
print("platoon", PLATOON)
