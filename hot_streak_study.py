"""
Hot-streak study: does a hitter's recent Statcast "form" predict his next stretch
of plate appearances beyond his established level?

Usage: python3 hot_streak_study.py DATA_DIR OUT_DIR [--quick]

DATA_DIR holds monthly Statcast files (parquet or csv.gz) from fetch_statcast.py.

Two frames:
  A. "Is it real?"  Deviation of the recent window from the player's own in-season
     level (leave-out mean of the rest of his season) -> does it persist into the
     next window?  Benchmarked against a null where each player's game order is
     shuffled within the season (same talent, no streak structure).
  B. "Is it actionable?"  Using only information available at the time (prior-only
     baseline), how many points of next-window wOBA does recent form add, and how
     does that compare with the platoon advantage?
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

RNG = np.random.default_rng(7)

SWING = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play"}
WHIFF = {"swinging_strike", "swinging_strike_blocked"}


# ----------------------------------------------------------------------------- load
def load(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")) +
                   glob.glob(os.path.join(data_dir, "*.csv.gz")))
    if not files:
        sys.exit(f"no data files in {data_dir}")
    dfs = [pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["game_type"] == "R"].copy()
    # Savant/pybaseball return pandas nullable dtypes; convert to plain numpy (NA -> NaN)
    for c in df.columns:
        dt = str(df[c].dtype)
        if dt in ("Int8", "Int16", "Int32", "Int64", "Float32", "Float64", "UInt8", "UInt16", "UInt32", "UInt64"):
            df[c] = df[c].astype("float64")
        elif dt in ("string", "str") or dt.startswith("string"):
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    df["batter"] = df["batter"].astype("int64")
    df["pitcher"] = df["pitcher"].astype("int64")
    df["game_pk"] = df["game_pk"].astype("int64")
    df["game_year"] = df["game_year"].astype("int64")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    return df


# ------------------------------------------------------------------- pitch features
def pitch_features(df):
    d = df
    desc = d["description"].astype(str)
    d["swing"] = desc.isin(SWING).astype(np.int8)
    d["whiff"] = desc.isin(WHIFF).astype(np.int8)
    z = pd.to_numeric(d["zone"], errors="coerce")
    d["inzone"] = z.between(1, 9).astype(np.int8)
    d["outzone"] = z.between(11, 14).astype(np.int8)
    d["z_swing"] = d["swing"] * d["inzone"]
    d["o_swing"] = d["swing"] * d["outzone"]

    d["pa"] = (pd.to_numeric(d["woba_denom"], errors="coerce") == 1).astype(np.int8)
    wv = pd.to_numeric(d["woba_value"], errors="coerce").fillna(0.0)
    d["woba_num"] = np.where(d["pa"] == 1, wv, 0.0)
    ev = pd.to_numeric(d["launch_speed"], errors="coerce")
    xw = pd.to_numeric(d["estimated_woba_using_speedangle"], errors="coerce")
    is_bip = (d["type"] == "X") & ev.notna()
    d["bip"] = (is_bip & (d["pa"] == 1)).astype(np.int8)
    d["xwoba_num"] = np.where(d["pa"] == 1, np.where(is_bip & xw.notna(), xw, wv), 0.0)
    d["xwobacon_num"] = np.where(d["bip"] == 1, xw.fillna(wv), 0.0)
    d["ev_num"] = np.where(d["bip"] == 1, ev, 0.0)
    d["hard"] = ((d["bip"] == 1) & (ev >= 95)).astype(np.int8)
    d["barrel"] = ((d["bip"] == 1) & (pd.to_numeric(d["launch_speed_angle"], errors="coerce") == 6)).astype(np.int8)
    ev_ = d["events"].astype(str)
    d["k"] = ((d["pa"] == 1) & ev_.isin(["strikeout", "strikeout_double_play"])).astype(np.int8)
    d["bb"] = ((d["pa"] == 1) & (ev_ == "walk")).astype(np.int8)

    # competitive swings (Savant-style): drop a hitter's slowest 10% of tracked swings
    # in that season unless the ball was hit 90+ mph
    bs = pd.to_numeric(d["bat_speed"], errors="coerce")
    sl = pd.to_numeric(d["swing_length"], errors="coerce")
    p10 = bs.groupby([d["batter"], d["game_year"]]).transform(lambda s: s.quantile(0.10))
    comp = bs.notna() & sl.notna() & ((bs >= p10) | (ev >= 90))
    d["comp"] = comp.astype(np.int8)
    d["bs_num"] = np.where(comp, bs, 0.0)
    d["sl_num"] = np.where(comp, sl, 0.0)
    d["fast"] = (comp & (bs >= 75)).astype(np.int8)
    plate_speed = pd.to_numeric(d["release_speed"], errors="coerce") * 0.92
    max_ev = 1.23 * bs + 0.23 * plate_speed
    ratio = ev / max_ev
    contact = comp & is_bip
    d["sq_up"] = (contact & (ratio >= 0.80)).astype(np.int8)
    d["blast"] = (contact & (ratio * 100 + bs >= 164)).astype(np.int8)

    # platoon: same-handed matchup (switch hitters always get the platoon edge)
    d["same_hand"] = (d["stand"] == d["p_throws"]).astype(np.int8)
    return d


# ------------------------------------------------------------ context adjustments
def context(d):
    pa = d[d["pa"] == 1]
    lg_x = pa["xwoba_num"].mean()
    # opposing pitcher quality: regressed xwOBA allowed that season
    g = pa.groupby(["pitcher", "game_year"])["xwoba_num"].agg(["sum", "count"])
    g["opp_q"] = (g["sum"] + 150 * lg_x) / (g["count"] + 150) - lg_x
    d = d.merge(g["opp_q"].reset_index(), on=["pitcher", "game_year"], how="left")
    d["opp_q"] = d["opp_q"].fillna(0.0)
    # park: wOBA in a team's home games minus its road games (same teams involved), half-regressed
    home = pa.groupby(["home_team", "game_year"])["woba_num"].mean()
    road = pd.concat([pa.assign(t=pa["away_team"]), ]).groupby(["t", "game_year"])["woba_num"].mean()
    road.index.names = ["home_team", "game_year"]
    pf = (0.5 * (home - road)).rename("park").reset_index()
    d = d.merge(pf, on=["home_team", "game_year"], how="left")
    d["park"] = d["park"].fillna(0.0)
    d["opp_q_num"] = d["opp_q"] * d["pa"]
    d["park_num"] = d["park"] * d["pa"]
    d["same_num"] = d["same_hand"] * d["pa"]
    return d


SUM_COLS = ["pa", "woba_num", "xwoba_num", "k", "bb", "swing", "whiff", "inzone", "outzone",
            "z_swing", "o_swing", "bip", "ev_num", "hard", "barrel", "xwobacon_num",
            "comp", "bs_num", "sl_num", "fast", "sq_up", "blast",
            "opp_q_num", "park_num", "same_num"]
PITCHES = "n_pitch"

# metric: (numerator, denominator, regression constant in denominator units, label)
METRICS = {
    "woba":       ("woba_num", "pa", 300, "wOBA (results)"),
    "xwoba":      ("xwoba_num", "pa", 150, "xwOBA"),
    "k_rate":     ("k", "pa", 60, "Strikeout rate"),
    "bb_rate":    ("bb", "pa", 120, "Walk rate"),
    "bat_speed":  ("bs_num", "comp", 15, "Bat speed"),
    "swing_len":  ("sl_num", "comp", 15, "Swing length"),
    "fast_swing": ("fast", "comp", 40, "Fast-swing rate (75+ mph)"),
    "squared_up": ("sq_up", "comp", 150, "Squared-up rate"),
    "blast":      ("blast", "comp", 150, "Blast rate"),
    "chase":      ("o_swing", "outzone", 120, "Chase rate"),
    "z_swing":    ("z_swing", "inzone", 120, "In-zone swing rate"),
    "whiff":      ("whiff", "swing", 80, "Whiff rate"),
    "ev":         ("ev_num", "bip", 40, "Avg exit velocity"),
    "hard_hit":   ("hard", "bip", 60, "Hard-hit rate"),
    "barrel":     ("barrel", "bip", 60, "Barrel rate"),
    "xwobacon":   ("xwobacon_num", "bip", 120, "xwOBA on contact"),
}
PROCESS = ["bat_speed", "swing_len", "fast_swing", "squared_up", "blast", "chase", "z_swing",
           "whiff", "ev", "hard_hit", "barrel", "xwobacon", "xwoba", "k_rate", "bb_rate"]


def batter_games(d):
    d = d.copy()
    d[PITCHES] = 1
    agg = d.groupby(["batter", "game_year", "game_date", "game_pk"], sort=False)[SUM_COLS + [PITCHES]].sum()
    bg = agg.reset_index().sort_values(["batter", "game_date", "game_pk"]).reset_index(drop=True)
    bg = bg[bg["pa"] > 0].reset_index(drop=True)
    return bg


def league_means(bg):
    tot = bg[SUM_COLS].sum()
    return {m: tot[n] / tot[dn] for m, (n, dn, _, _) in METRICS.items()}


def rates(S, lg, regress=False):
    out = {}
    for m, (n, dn, k, _) in METRICS.items():
        num, den = S[n], S[dn]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[m] = (num + k * lg[m]) / (den + k) if regress else np.where(den > 0, num / den, np.nan)
        out[m + "__n"] = den
    return out


# --------------------------------------------------------------------- windows
def build_windows(bg, recent_pa, fwd_pa, lg, shuffle=False):
    """One row per (batter, game) decision point with recent / forward / baseline / oracle."""
    bg = bg.copy()
    if shuffle:  # destroy within-season order, keep each player's season totals
        bg["_r"] = RNG.random(len(bg))
        bg = bg.sort_values(["batter", "game_year", "_r"]).reset_index(drop=True)
    X = bg[SUM_COLS].to_numpy(float)
    C = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, axis=0)])
    col = {c: i for i, c in enumerate(SUM_COLS)}
    cpa = C[:, col["pa"]]
    n = len(bg)
    idx = np.arange(n)
    key = bg["batter"].astype(str) + "_" + bg["game_year"].astype(str)
    s_start = bg.groupby(key, sort=False).cumcount().to_numpy()
    s_start = idx - s_start
    s_end = s_start + bg.groupby(key, sort=False)["pa"].transform("size").to_numpy()
    b_start = idx - bg.groupby("batter", sort=False).cumcount().to_numpy()

    j = np.searchsorted(cpa, cpa[idx] - recent_pa, side="right") - 1
    k = np.searchsorted(cpa, cpa[idx] + fwd_pa, side="left")
    ok = (j >= s_start) & (k <= s_end)
    i, j, k = idx[ok], j[ok], k[ok]
    ss, se, bs = s_start[ok], s_end[ok], b_start[ok]

    def S(a, b):
        return {c: C[b, col[c]] - C[a, col[c]] for c in SUM_COLS}

    rec, fwd = S(j, i), S(i, k)
    prior = S(bs, j)                         # everything before the recent window
    season = S(ss, se)
    oracle = {c: season[c] - rec[c] - fwd[c] for c in SUM_COLS}

    cols = {
        "batter": bg["batter"].to_numpy()[i], "game_year": bg["game_year"].to_numpy()[i],
        "game_date": bg["game_date"].to_numpy()[i],
        "season_pa": season["pa"], "prior_pa": prior["pa"],
        "fwd_pa": fwd["pa"], "rec_pa": rec["pa"],
        "fwd_oppq": fwd["opp_q_num"] / fwd["pa"], "fwd_park": fwd["park_num"] / fwd["pa"],
        "fwd_same": fwd["same_num"] / fwd["pa"],
    }
    std = S(ss, j)                           # season-to-date, before the recent window
    for tag, Sx, reg in [("rec", rec, False), ("fwd", fwd, False), ("prior", prior, True),
                         ("orc", oracle, False), ("std", std, True)]:
        r = rates(Sx, lg, regress=reg)
        for m in METRICS:
            cols[f"{tag}_{m}"] = r[m]
            cols[f"{tag}_{m}__n"] = r[m + "__n"]
    return pd.DataFrame(cols)


# ----------------------------------------------------------------- estimation
def ols_cluster(y, X, groups):
    X = sm.add_constant(X, has_constant="add")
    m = sm.OLS(y, X, missing="drop").fit(cov_type="cluster", cov_kwds={"groups": groups})
    return m


MIN_N = {"pa": 1, "comp": 30, "outzone": 40, "inzone": 40, "swing": 30, "bip": 15}


def frame_a(W, W0):
    """Persistence of each metric's deviation from in-season level, vs shuffled null."""
    rows = []
    base = (W["season_pa"] >= 300)
    base0 = (W0["season_pa"] >= 300)
    for m, (_, dn, _, lab) in METRICS.items():
        res = {}
        for tag, D, b in [("actual", W, base), ("null", W0, base0)]:
            ok = b & (D[f"rec_{m}__n"] >= MIN_N[dn]) & (D[f"fwd_{m}__n"] >= MIN_N[dn]) & (D[f"orc_{m}__n"] >= 3 * MIN_N[dn])
            D = D[ok]
            x = D[f"rec_{m}"] - D[f"orc_{m}"]
            y = D[f"fwd_{m}"] - D[f"orc_{m}"]
            fit = ols_cluster(y.to_numpy(), x.to_numpy()[:, None], D["batter"].to_numpy())
            res[tag] = (fit.params[1], fit.bse[1], len(D), np.corrcoef(x, y)[0, 1], x.std())
        a, nl = res["actual"], res["null"]
        rows.append({"metric": m, "label": lab,
                     "persistence": a[0] - nl[0], "se": float(np.hypot(a[1], nl[1])),
                     "slope_actual": a[0], "slope_null": nl[0],
                     "corr_actual": a[3], "corr_null": nl[3], "sd_recent_dev": a[4], "n": a[2]})
    return pd.DataFrame(rows).sort_values("persistence", ascending=False)


def cross_predict(W, W0, target="woba"):
    """Does a +1 SD recent deviation in metric m predict next-window wOBA (vs in-season level)?"""
    rows = []
    for tag, D in [("actual", W), ("null", W0)]:
        D = D[D["season_pa"] >= 300]
        y = (D[f"fwd_{target}"] - D[f"orc_{target}"]) * 1000
        for m, (_, dn, _, lab) in METRICS.items():
            ok = (D[f"rec_{m}__n"] >= MIN_N[dn]) & (D[f"orc_{m}__n"] >= 3 * MIN_N[dn])
            x = (D[f"rec_{m}"] - D[f"orc_{m}"])[ok]
            z = (x / x.std()).to_numpy()
            fit = ols_cluster(y[ok].to_numpy(), z[:, None], D["batter"][ok].to_numpy())
            rows.append({"metric": m, "label": lab, "frame": tag, "pts_per_sd": fit.params[1], "se": fit.bse[1]})
    t = pd.DataFrame(rows).pivot_table(index=["metric", "label"], columns="frame", values=["pts_per_sd", "se"])
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    t = t.reset_index()
    t["effect"] = t["pts_per_sd_actual"] - t["pts_per_sd_null"]
    t["effect_se"] = np.hypot(t["se_actual"], t["se_null"])
    return t.sort_values("effect", ascending=False)


def frame_b(W, lg, target="woba"):
    """Realistic forecast: prior-only baseline + recent form. Out-of-sample check on last season."""
    D = W[(W["prior_pa"] >= 200)].copy()
    feats_base = ["prior_woba", "prior_xwoba", "prior_k_rate", "prior_bb_rate", "fwd_oppq", "fwd_park", "fwd_same"]
    # recent-form terms: deviation from the player's own prior baseline, recent sample
    # shrunk toward zero so tiny samples don't dominate
    recent_terms = []
    for m in ["woba"] + PROCESS:
        _, dn, kreg, _ = METRICS[m]
        n = D[f"rec_{m}__n"]
        shrink = n / (n + kreg)
        dev = (D[f"rec_{m}"] - D[f"prior_{m}"]).fillna(0.0) * shrink
        D[f"dev_{m}"] = dev
        recent_terms.append(f"dev_{m}")
    y = D[f"fwd_{target}"] * 1000
    last = D["game_year"].max()
    train, test = D["game_year"] < last, D["game_year"] == last

    def fit_eval(cols):
        Xtr, Xte = D.loc[train, cols], D.loc[test, cols]
        f = sm.WLS(y[train], sm.add_constant(Xtr), weights=D.loc[train, "fwd_pa"]).fit()
        pred = f.predict(sm.add_constant(Xte, has_constant="add"))
        w = D.loc[test, "fwd_pa"]
        mse = np.average((y[test] - pred) ** 2, weights=w)
        full = sm.WLS(y, sm.add_constant(D[cols]), weights=D["fwd_pa"]).fit(
            cov_type="cluster", cov_kwds={"groups": D["batter"]})
        return f, full, mse, pred

    base_f, base_full, base_mse, base_pred = fit_eval(feats_base)
    models = {"baseline": {"oos_mse": base_mse}}
    for name, cols in [("results_streak", feats_base + ["dev_woba"]),
                       ("process_form", feats_base + [f"dev_{m}" for m in PROCESS]),
                       ("everything", feats_base + recent_terms)]:
        f, full, mse, pred = fit_eval(cols)
        # how far does recent form move the forecast? spread of the recent-form component,
        # using coefficients fit on earlier seasons applied to the held-out season
        comp = sum(f.params[c] * D.loc[test, c] for c in cols if c.startswith("dev_"))
        q = np.quantile(comp, [0.1, 0.5, 0.9])
        resid = (y[test] - base_pred)
        qd = pd.qcut(comp.rank(method="first"), 10, labels=False)
        tmp = pd.DataFrame({"q": qd, "r": resid, "w": D.loc[test, "fwd_pa"], "c": comp, "b": D.loc[test, "batter"]})
        oos_dec = tmp.groupby("q").apply(lambda s: pd.Series({
            "predicted_bump": s["c"].mean(),
            "actual_bump": np.average(s["r"], weights=s["w"]),
            "se": s["r"].std() / np.sqrt(s["b"].nunique())}), include_groups=False).reset_index()
        models[name] = {
            "oos_deciles": oos_dec.to_dict(orient="list"),
            "oos_mse": mse, "oos_rmse_gain_pts": float(np.sqrt(base_mse) - np.sqrt(mse)),
            "oos_r2_gain": float((base_mse - mse) / np.var(y[test])),
            "form_component_p10": q[0], "form_component_p50": q[1], "form_component_p90": q[2],
            "form_component_sd": float(comp.std()),
            "coefs": {c: [float(full.params[c]), float(full.bse[c])] for c in cols if c.startswith("dev_")},
        }
    # decile table: realised next-window wOBA minus baseline-model forecast, by recent-form decile
    D["resid_base"] = y - base_full.predict(sm.add_constant(D[feats_base]))
    dec = {}
    for m in ["woba", "xwoba", "bat_speed", "ev", "chase", "whiff", "blast"]:
        q = pd.qcut(D[f"dev_{m}"].rank(method="first"), 10, labels=False)
        g = D.groupby(q).apply(lambda s: pd.Series({
            "resid": np.average(s["resid_base"], weights=s["fwd_pa"]),
            "se": s["resid_base"].std() / np.sqrt(s["batter"].nunique()),
            "dev_mean": s[f"dev_{m}"].mean(),
            "raw_dev_mean": (s[f"rec_{m}"] - s[f"prior_{m}"]).mean()}), include_groups=False)
        dec[m] = g.reset_index().rename(columns={f"dev_{m}": "decile"}).to_dict(orient="list")
    return models, dec, D


def platoon(d):
    pa = d[d["pa"] == 1]
    out = {}
    for st in ["L", "R"]:
        s = pa[pa["stand"] == st]
        same = s[s["same_hand"] == 1]["woba_num"].mean()
        opp = s[s["same_hand"] == 0]["woba_num"].mean()
        out[f"{st}HB_league_split_pts"] = float((opp - same) * 1000)
    # individual regressed splits (Tango-style regression toward the league split for the handedness)
    rows = []
    for (b, st), s in pa[pa["stand"].isin(["L", "R"])].groupby(["batter", "stand"]):
        ns = int(s["same_hand"].sum())
        no = len(s) - ns
        if ns < 60 or no < 60:
            continue
        obs = s.loc[s["same_hand"] == 0, "woba_num"].mean() - s.loc[s["same_hand"] == 1, "woba_num"].mean()
        lgs = out[f"{st}HB_league_split_pts"] / 1000
        k = 1000 if st == "L" else 2200  # PA vs same-hand pitchers needed for 50% weight
        rows.append({"batter": b, "stand": st, "pa_same": ns,
                     "reg_split_pts": 1000 * (lgs + (obs - lgs) * ns / (ns + k))})
    r = pd.DataFrame(rows)
    for st in ["L", "R"]:
        q = r[r["stand"] == st]["reg_split_pts"].quantile([0.1, 0.5, 0.9])
        out[f"{st}HB_individual_p10_p50_p90"] = [float(v) for v in q]
    return out


# ------------------------------------------------------------------------ main
def main():
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    print("loading ...", flush=True)
    d = load(data_dir)
    print(f"  {len(d):,} pitches, seasons {sorted(d['game_year'].unique())}")
    d = pitch_features(d)
    d = context(d)
    bg = batter_games(d)
    lg = league_means(bg)
    print(f"  {len(bg):,} batter-games, {bg['batter'].nunique():,} hitters")
    summary = {"n_pitches": int(len(d)), "n_batter_games": int(len(bg)),
               "n_hitters": int(bg["batter"].nunique()),
               "seasons": [int(s) for s in sorted(d["game_year"].unique())],
               "league": lg, "platoon": platoon(d)}

    results = {}
    for rp, fp in [(50, 50), (25, 25), (100, 50), (50, 25)]:
        tag = f"rec{rp}_fwd{fp}"
        print(f"windows {tag} ...", flush=True)
        W = build_windows(bg, rp, fp, lg)
        W0 = build_windows(bg, rp, fp, lg, shuffle=True)
        pa_ = frame_a(W, W0)
        cp = cross_predict(W, W0, "woba")
        cpx = cross_predict(W, W0, "xwoba")
        models, dec, _ = frame_b(W, lg, "woba")
        models_x, dec_x, _ = frame_b(W, lg, "xwoba")
        pa_.to_csv(os.path.join(out_dir, f"persistence_{tag}.csv"), index=False)
        cp.to_csv(os.path.join(out_dir, f"cross_woba_{tag}.csv"), index=False)
        cpx.to_csv(os.path.join(out_dir, f"cross_xwoba_{tag}.csv"), index=False)
        results[tag] = {"n_decision_points": int(len(W)), "persistence": pa_.to_dict(orient="records"),
                        "cross_woba": cp.to_dict(orient="records"),
                        "cross_xwoba": cpx.to_dict(orient="records"),
                        "models_woba": models, "models_xwoba": models_x,
                        "deciles_woba": dec, "deciles_xwoba": dec_x}
        print(pa_[["metric", "persistence", "se", "slope_actual", "slope_null"]].round(3).to_string(index=False))
        print(cp[["metric", "effect", "effect_se"]].round(2).to_string(index=False))
        for k_, v in models.items():
            print(" ", k_, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "coefs"})
    summary["results"] = results
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, default=lambda o: o.item() if hasattr(o, "item") else str(o), indent=1)
    print("platoon:", summary["platoon"])
    print("done")


if __name__ == "__main__":
    main()
