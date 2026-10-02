"""Full-paper analyses A1-A10 (plan: PLAN_2026-09-30.md, written before any of these were run).

python3 analysis3.py DATA_DIR PUBLIC_DIR OUT_DIR [A1 A2 ...]      (no list = all)

Each analysis writes OUT_DIR/a<k>.json; make_paper_results.py merges them into paper_results.json.
Public inputs (PUBLIC_DIR): il_transactions_hitters_2024_2026.txt, marcel_woba_2024_2026.txt,
postseason_woba_2024_2025.txt, all rebuilt by fetch_public.py from the MLB Stats API.
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

import common3 as C
import hot_streak_study as H

DATA, PUB, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
WANT = set(sys.argv[4:]) or {f"A{k}" for k in range(1, 11)}
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "_cache")
os.makedirs(CACHE, exist_ok=True)
SEED = 101
HEAD = ["bat_speed", "swing_len", "fast_swing", "z_swing", "chase", "xwoba", "woba"]
WOBA_SCALE = 1.24


def dump(name, obj):
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"wrote {name}.json", flush=True)


def cached(name, fn):
    p = os.path.join(CACHE, name + ".pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    v = fn()
    pickle.dump(v, open(p, "wb"))
    return v


def panel(adjusted=True, comp=True):
    def build():
        d = C.pitches(DATA, adjusted=adjusted, comp=comp)
        d["onb"] = ((d["pa"] == 1) & (pd.to_numeric(d["woba_value"], errors="coerce").fillna(0) > 0)).astype(np.int8)
        d["hr"] = ((d["pa"] == 1) & (d["events"].astype(str) == "home_run")).astype(np.int8)
        d["woba_same_num"] = d["woba_num"] * d["same_hand"]
        d[C.H.PITCHES] = 1
        keys = ["batter", "game_year", "game_date", "game_pk"]
        cols = C.SUM_COLS + ["onb", "hr", "woba_same_num", C.H.PITCHES]
        bg = d.groupby(keys, sort=False)[cols].sum().reset_index()
        bg = bg.sort_values(["batter", "game_date", "game_pk"]).reset_index(drop=True)
        bg = bg[bg["pa"] > 0].reset_index(drop=True)
        return bg, H.league_means(bg)
    return cached(f"bg_adj{int(adjusted)}_comp{int(comp)}", build)


def season_bounds(bg):
    g = bg.groupby("game_year")["game_date"].agg(["min", "max"])
    return {int(y): (int(np.datetime64(r["min"], "D").astype(np.int64)), int(np.datetime64(r["max"], "D").astype(np.int64)))
            for y, r in g.iterrows()}


def marcel():
    rows = []
    for tok in open(os.path.join(PUB, "marcel_woba_2024_2026.txt")).read().split():
        b, y, w, r, pa, age = tok.split(":")
        rows.append({"batter": int(b), "game_year": 2020 + int(y), "marcel": int(w) / 1000, "marcel_rel": int(r) / 100,
                     "pa_prev": int(pa), "age": float(age) if age else np.nan})
    M = pd.DataFrame(rows)
    M["age"] = M["age"].fillna(M["age"].median())
    return M


FG_BAT = os.environ.get("FG_BAT")            # FanGraphs position-player season lines (xlsx/parquet); not redistributed
SHARED = os.environ.get("SHARED_STATCAST")   # Statcast_Shared/data/statcast_season (full-width cache)


def fg_projection():
    """Marcel-style projected wRC+ from FanGraphs season lines (5/4/3 by PA, 1,200 PA of 100, Marcel age factor),
    plus prior-season FanGraphs WAR and PA. None when FG_BAT is not set."""
    if not FG_BAT:
        return None
    X = pd.read_parquet(FG_BAT) if FG_BAT.endswith(".parquet") else pd.read_excel(FG_BAT)
    X = X[["Season", "MLBAMID", "PA", "wRC+", "WAR", "Age"]].dropna(subset=["MLBAMID"])
    X["MLBAMID"] = X["MLBAMID"].astype(int)
    X["wrc"] = X["wRC+"].fillna(100.0)
    rows = []
    ids = X["MLBAMID"].unique()
    by = {k: g.set_index("Season") for k, g in X.groupby("MLBAMID")}
    for b in ids:
        g = by[b]
        for Y in (2024, 2025, 2026):
            num = den = 0.0
            age = None
            for k, w in zip((1, 2, 3), (5, 4, 3)):
                if Y - k in g.index:
                    r = g.loc[Y - k]
                    num += w * r["PA"] * r["wrc"]; den += w * r["PA"]
                    if age is None and pd.notna(r["Age"]):
                        age = r["Age"] + k
            proj = (num + 1200 * 100) / (den + 1200)
            if age is not None:
                proj = 100 + (proj - 100) + proj * ((0.006 * (29 - age)) if age < 29 else (-0.003 * (age - 29)))
            prev = g.loc[Y - 1] if Y - 1 in g.index else None
            rows.append({"batter": int(b), "game_year": Y, "fg_proj": proj, "fg_rel": den / (den + 1200),
                         "fg_age": age if age is not None else np.nan,
                         "war_prev": float(prev["WAR"]) if prev is not None else 0.0,
                         "fg_pa_prev": float(prev["PA"]) if prev is not None else 0.0})
    F = pd.DataFrame(rows)
    F["fg_age"] = F["fg_age"].fillna(F["fg_age"].median())
    return F


def il_events():
    P, A = [], []
    for line in open(os.path.join(PUB, "il_transactions_hitters_2024_2026.txt")):
        b, evs = line.strip().split(":")
        for e in evs.split():
            dt = pd.Timestamp("20" + e[:6])
            if e[6] == "P":
                P.append({"batter": int(b), "start": dt, "family": {"O": "oblique", "H": "hamstring", "W": "hand_wrist"}.get(e[7], "other")})
            else:
                A.append({"batter": int(b), "end": dt})
    return pd.DataFrame(P), pd.DataFrame(A)


def forecast_frame(W, min_prior=200):
    D = C.prep(W, min_prior)
    resid, base = C.resid_on(D, "woba", C.STRONG_BASE)
    return D, resid, base


# =========================================================================================== A1
def A1():
    bg, lg = panel()
    K = 10
    bg0s = [C.shuffle_games(bg, SEED + i) for i in range(K)]
    MET = ["bat_speed", "swing_len", "fast_swing", "z_swing", "woba"]
    out = {"horizons": {}, "n_shuffles": K}
    for fwd in [25, 50, 100, 150, 200]:
        W = C.windows(bg, 50, fwd, lg)
        out["horizons"][fwd] = C.persistence(W, [C.windows(b0, 50, fwd, lg) for b0 in bg0s], MET)
        print("A1 fwd", fwd, {m: round(v["persistence"], 3) for m, v in out["horizons"][fwd].items()}, flush=True)
    # same decision points at every horizon (points that have 200 PA ahead) so the curve is not a sample change
    keep = set(C.windows(bg, 50, 200, lg)["row"])
    keep0 = [set(C.windows(b0, 50, 200, lg)["row"]) for b0 in bg0s]
    out["horizons_fixed_sample"] = {}
    for fwd in [25, 50, 100, 150, 200]:
        W = C.windows(bg, 50, fwd, lg); W = W[W["row"].isin(keep)]
        W0s = []
        for b0, k0 in zip(bg0s, keep0):
            x = C.windows(b0, 50, fwd, lg); W0s.append(x[x["row"].isin(k0)])
        out["horizons_fixed_sample"][fwd] = C.persistence(W, W0s, MET)
        print("A1 fixed fwd", fwd, {m: round(v["persistence"], 3) for m, v in out["horizons_fixed_sample"][fwd].items()}, flush=True)

    # non-overlapping 50-PA blocks: lag-1 autocorrelation of block-mean bat speed and wOBA (deviation from season mean)
    def blocks(b):
        b = b.sort_values(["batter", "game_date", "game_pk"]).copy()
        b["cpa"] = b.groupby(["batter", "game_year"])["pa"].cumsum() - b["pa"]
        b["blk"] = (b["cpa"] // 50).astype(int)
        g = b.groupby(["batter", "game_year", "blk"])[["pa", "comp", "bs_num", "woba_num"]].sum().reset_index()
        g = g[g["pa"] >= 50]
        sp = g.groupby(["batter", "game_year"])["pa"].transform("sum")
        g = g[sp >= 300]
        res = {}
        for m, num, den in [("bat_speed", "bs_num", "comp"), ("woba", "woba_num", "pa")]:
            h = g[g[den] >= (30 if den == "comp" else 1)].copy()
            h["y"] = h[num] / h[den]
            h["y"] = h["y"] - h.groupby(["batter", "game_year"])["y"].transform("mean")
            h["yl"] = h.groupby(["batter", "game_year"])["y"].shift(1)
            h = h.dropna(subset=["yl"])
            res[m] = float(np.corrcoef(h["y"], h["yl"])[0, 1])
        return res
    bl = blocks(bg)
    bl0 = [blocks(b0) for b0 in bg0s]
    out["block_lag1"] = {m: {"actual": bl[m], "shuffled_mean": float(np.mean([x[m] for x in bl0]))} for m in bl}
    print("A1 blocks", out["block_lag1"], flush=True)

    # changepoint removal
    def changepoint_adjust(b):
        b = b.copy()
        g = b["comp"] >= 5
        lvl = np.zeros(len(b))
        info = []
        for (bat, yr), idx in b.groupby(["batter", "game_year"]).groups.items():
            idx = np.asarray(idx)
            q = idx[g.to_numpy()[idx]]
            if len(q) < 40:
                continue
            y = b.loc[q, "bs_num"].to_numpy() / b.loc[q, "comp"].to_numpy()
            w = b.loc[q, "comp"].to_numpy().astype(float)
            cw, cy, cyy = np.cumsum(w), np.cumsum(w * y), np.cumsum(w * y * y)
            ks = np.arange(20, len(q) - 20 + 1)
            w1, s1, ss1 = cw[ks - 1], cy[ks - 1], cyy[ks - 1]
            w2, s2, ss2 = cw[-1] - w1, cy[-1] - s1, cyy[-1] - ss1
            sse = (ss1 - s1 * s1 / w1) + (ss2 - s2 * s2 / w2)
            bk = int(ks[np.argmin(sse)])
            m1 = cy[bk - 1] / cw[bk - 1]
            m2 = (cy[-1] - cy[bk - 1]) / (cw[-1] - cw[bk - 1])
            split_row = q[bk]                       # first game of segment 2
            seg2 = idx >= split_row
            lvl[idx] = np.where(seg2, m2, m1)
            info.append(abs(m2 - m1))
        b["bs_num"] = b["bs_num"] - b["comp"] * lvl
        return b, np.array(info)

    bgc, jumps = changepoint_adjust(bg)
    adj0 = [changepoint_adjust(b0) for b0 in bg0s]
    jumps0 = np.concatenate([a[1] for a in adj0])
    Wb, Wa = C.windows(bg, 50, 50, lg), C.windows(bgc, 50, 50, lg)
    before = C.persistence(Wb, [C.windows(b0, 50, 50, lg) for b0 in bg0s], ["bat_speed"])["bat_speed"]
    after = C.persistence(Wa, [C.windows(a[0], 50, 50, lg) for a in adj0], ["bat_speed"])["bat_speed"]
    out["changepoint"] = {"before": before, "after": after, "share_surviving": after["persistence"] / before["persistence"],
                          "n_hitter_seasons_fitted": int(len(jumps)), "median_jump_mph": float(np.median(jumps)),
                          "median_jump_mph_shuffled": float(np.median(jumps0)),
                          "share_jump_over_1mph": float((jumps > 1).mean()), "share_jump_over_1mph_shuffled": float((jumps0 > 1).mean())}

    def ar1(b):
        g = b[b["comp"] >= 5].copy()
        g["y"] = g["bs_num"] / g["comp"]
        g["y"] = g["y"] - g.groupby(["batter", "game_year"])["y"].transform("mean")
        g["ylag"] = g.groupby(["batter", "game_year"])["y"].shift(1)
        g = g.dropna(subset=["ylag"])
        f = C.ols_cl(g["y"].to_numpy(), g["ylag"].to_numpy(), g["batter"].to_numpy())
        return float(f.params[1]), float(f.bse[1])
    out["ar1_game_level"] = {"raw": ar1(bg), "raw_shuffled_mean": float(np.mean([ar1(b0)[0] for b0 in bg0s[:3]])),
                             "after_changepoint": ar1(bgc), "after_changepoint_shuffled_mean": float(np.mean([ar1(a[0])[0] for a in adj0[:3]]))}
    print("A1 changepoint", before["persistence"], after["persistence"], out["ar1_game_level"], flush=True)
    dump("a1", out)


# =========================================================================================== A2
def A2():
    bg, lg = panel()
    M = marcel()
    Wall = C.windows(bg, 50, 50, lg, at_risk=True)
    Dall = C.prep(Wall, 200)
    D = Dall[Dall["complete"]].copy()
    resid, base = C.resid_on(D, "woba", C.STRONG_BASE)
    prim, comp, coefs = C.form_summary(D, resid, C.SWING)
    D["form"] = comp
    qfull = pd.qcut(D["form"].rank(method="first"), 10, labels=False)
    out = {"primary": prim}

    # (i) IPW
    A = Dall.merge(M, on=["batter", "game_year"], how="left")
    A["marcel"] = A["marcel"].fillna(A["marcel"].median()); A["pa_prev"] = A["pa_prev"].fillna(0)
    A["age"] = A["age"].fillna(A["age"].median())
    A["doy"] = pd.to_datetime(A["game_date"]).dt.dayofyear
    A["z_bs"] = A["raw_bat_speed"].fillna(0) / A["raw_bat_speed"].std()
    A["z_w"] = A["raw_woba"].fillna(0) / A["raw_woba"].std()
    X = pd.DataFrame({"z_bs": A["z_bs"], "z_w": A["z_w"], "marcel": A["marcel"], "pa_prev": A["pa_prev"] / 100,
                      "age": A["age"], "doy": A["doy"], "doy2": (A["doy"] / 100) ** 2})
    FG = fg_projection()
    if FG is not None:
        A = A.merge(FG[["batter", "game_year", "war_prev", "fg_pa_prev"]], on=["batter", "game_year"], how="left")
        X["war_prev"] = A["war_prev"].fillna(0.0).to_numpy()
    lg_ = sm.Logit(A["complete"].astype(int), sm.add_constant(X)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": A["batter"]})
    p = lg_.predict(sm.add_constant(X))
    w = A["complete"].mean() / p
    lo, hi = np.quantile(w[A["complete"]], [0.01, 0.99])
    w = w.clip(lo, hi)
    A["w"] = w.to_numpy()
    wD = pd.Series(A.loc[A["complete"], "w"].to_numpy(), index=D.index)
    ipw = C.decile_effects(D, resid, q=qfull, weights=wD)
    out["ipw"] = {"top": ipw["bump"][9], "top_se": ipw["se"][9], "bottom": ipw["bump"][0], "bottom_se": ipw["se"][0],
                  "logit": {c: [float(lg_.params[c]), float(lg_.bse[c])] for c in X.columns},
                  "share_complete": float(A["complete"].mean()), "n_at_risk": int(len(A)),
                  "weight_range": [float(lo), float(hi)]}

    # (ii) forward window crosses the season boundary
    Wx = C.windows(bg, 50, 50, lg, cross_season=True)
    Dx = C.prep(Wx, 200)
    bx = C.STRONG_BASE + ["crosses"]
    rx, _ = C.resid_on(Dx, "woba", bx)
    sx, _, _ = C.form_summary(Dx, rx, C.SWING)
    sx["n_crossing"] = int(Dx["crosses"].sum())
    out["cross_season"] = sx

    # (iii) established regulars: >= 400 MLB PA the previous season
    Dm = D.merge(M[["batter", "game_year", "pa_prev"]], on=["batter", "game_year"], how="left").set_index(D.index)
    reg = Dm["pa_prev"].fillna(0) >= 400
    er = C.decile_effects(D[reg], resid[reg], q=qfull)
    out["regulars_400pa"] = {"top": er["bump"][9], "top_se": er["se"][9], "bottom": er["bump"][0], "bottom_se": er["se"][0],
                             "n": int(reg.sum()), "hitters": int(D.loc[reg, "batter"].nunique()),
                             "n_bottom_decile": er["n"][0]}
    if FG is not None:
        Df = D.merge(FG[["batter", "game_year", "fg_pa_prev"]], on=["batter", "game_year"], how="left").set_index(D.index)
        regf = Df["fg_pa_prev"].fillna(0) >= 400
        e2 = C.decile_effects(D[regf], resid[regf], q=qfull)
        out["regulars_400pa_fangraphs"] = {"top": e2["bump"][9], "top_se": e2["se"][9], "bottom": e2["bump"][0], "bottom_se": e2["se"][0],
                                           "n": int(regf.sum()), "hitters": int(D.loc[regf, "batter"].nunique())}
        out["ipw"]["includes_prior_war"] = True
    print("A2", {k: (round(v["bottom"], 2), round(v["top"], 2)) for k, v in out.items()}, flush=True)
    dump("a2", out)


# =========================================================================================== A3
def A3():
    bg, lg = panel()
    M = marcel()
    W = C.windows(bg, 50, 50, lg)
    out = {}
    ctx = ["std_woba", "std_xwoba", "std_bat_speed", "std_ev", "std_k_rate", "std_chase", "std_whiff", "fwd_oppq", "fwd_park", "fwd_same"]
    FGp = fg_projection()
    for tag, mp in [("primary_sample", 200), ("enlarged_sample", 0)]:
        D = C.prep(W, mp)
        D = D.merge(M, on=["batter", "game_year"], how="left").set_index(D.index)
        if FGp is not None:
            D = D.merge(FGp[["batter", "game_year", "fg_proj", "fg_rel", "fg_age"]], on=["batter", "game_year"], how="left").set_index(D.index)
            D["fg_proj"] = D["fg_proj"].fillna(100.0); D["fg_rel"] = D["fg_rel"].fillna(0.0); D["fg_age"] = D["fg_age"].fillna(D["fg_age"].median())
        has = D["marcel"].notna()
        D = D[has].copy()
        for c in ctx:
            D[c] = D[c].fillna(D[c].mean())
        res = {"n": int(len(D)), "hitters": int(D["batter"].nunique()), "share_with_marcel": float(has.mean())}
        specs = [("marcel", ["marcel", "marcel_rel", "age"] + ctx),
                 ("marcel_plus_history", C.STRONG_BASE + ["marcel", "marcel_rel", "age"]),
                 ("history_only", C.STRONG_BASE)]
        if FGp is not None:
            specs += [("fangraphs_marcel", ["fg_proj", "fg_rel", "fg_age"] + ctx),
                      ("fangraphs_plus_history", C.STRONG_BASE + ["fg_proj", "fg_rel", "fg_age"])]
        for bname, bcols in specs:
            r, f = C.resid_on(D, "woba", bcols)
            y = D["fwd_woba"] * 1000
            r2 = 1 - np.average(r ** 2, weights=D["fwd_pa"]) / np.average((y - np.average(y, weights=D["fwd_pa"])) ** 2, weights=D["fwd_pa"])
            s_sw, _, _ = C.form_summary(D, r, C.SWING)
            s_rs, _, _ = C.form_summary(D, r, ["dev_woba"])
            res[bname] = {"baseline_r2": float(r2), "swing_only": {k: s_sw[k] for k in ["top", "top_se", "bottom", "bottom_se", "oos_r2_gain"]},
                          "results_only": {k: s_rs[k] for k in ["top", "top_se", "bottom", "bottom_se", "oos_r2_gain"]}}
            if bname == "marcel":
                res[bname]["marcel_coef"] = [float(f.params["marcel"]), float(f.bse["marcel"])]
            print("A3", tag, bname, round(r2, 4), res[bname]["swing_only"], res[bname]["results_only"]["top"], flush=True)
        out[tag] = res
    dump("a3", out)


# =========================================================================================== A4
def A4():
    bg, lg = panel()
    bg0 = C.shuffle_games(bg, SEED)
    out = {}

    def thin(W, step):
        W = W.sort_values(["batter", "game_date"]).copy()
        keep = []
        last_b, last_y, nxt = None, None, -1
        # use the season-to-date PA (std) = prior within season; approximate by cumulative position of the decision
        W["pos"] = W["std_woba__n"] + W["rec_pa"]      # PA before the decision point within the season
        for b, y, p in zip(W["batter"], W["game_year"], W["pos"]):
            if b != last_b or y != last_y:
                last_b, last_y, nxt = b, y, -1
            if p >= nxt:
                keep.append(True); nxt = p + step
            else:
                keep.append(False)
        return W[np.array(keep)]

    for m in ["bat_speed", "woba"]:
        res = {}
        for step in [50, 100]:
            r = {}
            for tag, b in [("actual", bg), ("null", bg0)]:
                W = C.windows(b, 50, 50, lg)
                dn = C.METRICS[m][1]; mn = H.MIN_N[dn]
                W = W[(W["season_pa"] >= 300) & (W[f"rec_{m}__n"] >= mn) & (W[f"fwd_{m}__n"] >= mn) & (W[f"orc_{m}__n"] >= 3 * mn)]
                W = thin(W, step)
                W["x"] = W[f"rec_{m}"] - W[f"orc_{m}"]
                W["y"] = W[f"fwd_{m}"] - W[f"orc_{m}"]
                sl = []
                for bat, g in W.groupby("batter"):
                    if len(g) < 15 or g["x"].std() == 0:
                        continue
                    f = sm.OLS(g["y"].to_numpy(), sm.add_constant(g["x"].to_numpy())).fit()
                    sl.append((bat, f.params[1], f.bse[1], len(g)))
                S = pd.DataFrame(sl, columns=["batter", "b", "se", "n"])
                wgt = 1 / S["se"] ** 2
                mu = float(np.sum(wgt * S["b"]) / wgt.sum())
                Q = float(np.sum(wgt * (S["b"] - mu) ** 2))
                k = len(S)
                tau2_dl = max(0.0, (Q - (k - 1)) / (wgt.sum() - (wgt ** 2).sum() / wgt.sum()))
                tau2_mm = max(0.0, float(S["b"].var() - (S["se"] ** 2).mean()))
                med_se2 = float(np.median(S["se"] ** 2))
                shrink = tau2_dl / (tau2_dl + S["se"] ** 2)
                post = mu + shrink * (S["b"] - mu)
                post_sd = np.sqrt(tau2_dl * S["se"] ** 2 / (tau2_dl + S["se"] ** 2)) if tau2_dl > 0 else np.zeros(k)
                n_diff = int(np.sum((np.abs(post - mu) > 1.96 * post_sd) & (post_sd > 0)))
                pooled = C.ols_cl(W["y"].to_numpy(), W["x"].to_numpy(), W["batter"].to_numpy())
                r[tag] = {"hitters": k, "median_pairs": float(S["n"].median()), "pooled_slope": [float(pooled.params[1]), float(pooled.bse[1])],
                          "mean_slope_ivw": mu, "tau_dl": float(np.sqrt(tau2_dl)), "tau_mm": float(np.sqrt(tau2_mm)),
                          "median_se": float(np.sqrt(med_se2)), "reliability_median_hitter": float(tau2_dl / (tau2_dl + med_se2)) if tau2_dl > 0 else 0.0,
                          "Q": Q, "df": k - 1, "n_credibly_different": n_diff, "sd_raw_slopes": float(S["b"].std())}
                if step == 50 and tag == "actual":
                    try:
                        mm = sm.MixedLM(W["y"].to_numpy(), sm.add_constant(W["x"].to_numpy()), groups=W["batter"].to_numpy(),
                                        exog_re=sm.add_constant(W["x"].to_numpy())).fit(reml=True, method="lbfgs")
                        r[tag]["mixed_model_slope_sd"] = float(np.sqrt(max(np.asarray(mm.cov_re)[1, 1], 0)))
                        r[tag]["mixed_model_fixed_slope"] = float(np.asarray(mm.fe_params)[1])
                    except Exception as e:
                        r[tag]["mixed_model_error"] = str(e)
            res[f"step{step}"] = r
            print("A4", m, step, {t: (round(v["tau_dl"], 3), round(v["reliability_median_hitter"], 3), v["n_credibly_different"]) for t, v in r.items()}, flush=True)
        out[m] = res
    dump("a4", out)


# =========================================================================================== A5
PM = {  # metric: (numerator, denominator, min denominator, label)
    "ff_velo": ("ff_v", "ff", 15, "Four-seam velocity"),
    "whiff": ("whiff", "swing", 30, "Whiff rate"),
    "chase": ("o_swing", "outzone", 40, "Chase rate"),
    "zone": ("inzone", "n_pitch", 100, "Zone rate"),
    "xwoba_ag": ("xwoba_num", "pa", 20, "xwOBA against"),
    "kbb": ("kmb", "pa", 20, "K-BB%"),
    "woba_ag": ("woba_num", "pa", 20, "wOBA against"),
    "ff_spin": ("ff_s", "ff_sn", 15, "Four-seam spin rate"),
    "ff_ext": ("ff_e", "ff_en", 15, "Four-seam extension"),
    "ff_relz": ("ff_z", "ff_zn", 15, "Four-seam release height"),
    "ff_arm": ("ff_a", "ff_an", 15, "Four-seam arm angle (added in addendum B)"),
}


def pitcher_panel():
    def build():
        d = C.pitches(DATA, adjusted=False)
        d["ff"] = (d["pitch_type"] == "FF").astype(np.int8)
        d["ff_v"] = np.where(d["ff"] == 1, pd.to_numeric(d["release_speed"], errors="coerce").fillna(0), 0.0)
        d["ff"] = (d["ff"] == 1) & pd.to_numeric(d["release_speed"], errors="coerce").notna()
        d["kmb"] = d["k"].astype(int) - d["bb"].astype(int)
        d["n_pitch"] = 1
        cols = ["ff", "ff_v", "swing", "whiff", "o_swing", "outzone", "inzone", "n_pitch", "xwoba_num", "woba_num", "kmb", "pa"]
        if SHARED:
            ext = pd.concat([pd.read_parquet(os.path.join(SHARED, f"statcast_{y}.parquet"),
                                             columns=["game_type", "game_pk", "at_bat_number", "pitch_number", "release_spin_rate",
                                                      "release_extension", "release_pos_z", "arm_angle"]) for y in (2024, 2025, 2026)])
            ext = ext[ext["game_type"] == "R"].drop(columns="game_type").drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])
            for c in ["game_pk", "at_bat_number", "pitch_number"]:
                ext[c] = ext[c].astype("int64"); d[c] = d[c].astype("int64")
            n0 = len(d)
            d = d.merge(ext, on=["game_pk", "at_bat_number", "pitch_number"], how="left")
            assert len(d) == n0
            isff = d["pitch_type"] == "FF"
            if os.environ.get("PITCH_ADJUST", "1") == "1":
                # same park-season and week-of-season adjustment as the hitter swing metrics, computed on four-seamers
                wk = pd.to_datetime(d["game_date"]).dt.isocalendar().week.astype(int)
                for c in ["release_speed", "release_spin_rate", "release_extension", "release_pos_z", "arm_angle"]:
                    x = pd.to_numeric(d[c], errors="coerce").where(isff)
                    ym = x.groupby(d["game_year"]).transform("mean")
                    pk = x.groupby([d["home_team"], d["game_year"]]).transform("mean") - ym
                    wm = x.groupby([d["game_year"], wk]).transform("mean") - ym
                    d[c] = np.where(isff, pd.to_numeric(d[c], errors="coerce") - pk - wm, d[c])
                d["ff_v"] = np.where(isff & pd.to_numeric(d["release_speed"], errors="coerce").notna(), pd.to_numeric(d["release_speed"], errors="coerce"), 0.0)
            for src, a, b in [("release_spin_rate", "ff_s", "ff_sn"), ("release_extension", "ff_e", "ff_en"),
                              ("release_pos_z", "ff_z", "ff_zn"), ("arm_angle", "ff_a", "ff_an")]:
                v = pd.to_numeric(d[src], errors="coerce")
                ok = isff & v.notna()
                d[a] = np.where(ok, v, 0.0); d[b] = ok.astype(int)
                cols += [a, b]
        pg = d.groupby(["pitcher", "game_year", "game_date", "game_pk"], sort=False)[cols].sum().reset_index()
        return pg.sort_values(["pitcher", "game_date", "game_pk"]).reset_index(drop=True)
    return cached("pitcher_games" + ("_adj" if os.environ.get("PITCH_ADJUST", "1") == "1" else ""), build)


def pitcher_windows(pg, rec, fwd, unit="n_pitch"):
    cols = [c for c in pg.columns if c not in ("pitcher", "game_year", "game_date", "game_pk")]
    X = pg[cols].to_numpy(float)
    Cm = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, axis=0)])
    ci = {c: i for i, c in enumerate(cols)}
    cu = Cm[:, ci[unit]] if unit != "games" else np.arange(len(pg) + 1, dtype=float)
    idx = np.arange(len(pg))
    key = pg["pitcher"].astype(str) + "_" + pg["game_year"].astype(str)
    ss = idx - pg.groupby(key, sort=False).cumcount().to_numpy()
    se = ss + pg.groupby(key, sort=False)["pa"].transform("size").to_numpy()
    j = np.searchsorted(cu, cu[idx] - rec, side="right") - 1
    k = np.searchsorted(cu, cu[idx] + fwd, side="left")
    ok = (j >= ss) & (k <= se)
    i, j, k, ss, se = idx[ok], j[ok], k[ok], ss[ok], se[ok]
    S = lambda a, b: {c: Cm[b, ci[c]] - Cm[a, ci[c]] for c in cols}
    r, f, s = S(j, i), S(i, k), S(ss, se)
    o = {c: s[c] - r[c] - f[c] for c in cols}
    out = {"pitcher": pg["pitcher"].to_numpy()[i], "game_year": pg["game_year"].to_numpy()[i], "season_n": s["n_pitch"]}
    for tag, Sx in [("rec", r), ("fwd", f), ("orc", o)]:
        for m, (nu, de, mn, _) in PM.items():
            if nu not in Sx:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"{tag}_{m}"] = np.where(Sx[de] >= (mn if tag != "orc" else 3 * mn), Sx[nu] / np.maximum(Sx[de], 1), np.nan)
    return pd.DataFrame(out)


def shuffle_pitchers(pg, seed):
    rng = np.random.default_rng(seed)
    key = rng.random(len(pg))
    order = pd.DataFrame({"p": pg["pitcher"], "y": pg["game_year"], "k": key}).sort_values(["p", "y", "k"]).index.to_numpy()
    out = pg.copy()
    cols = [c for c in pg.columns if c not in ("pitcher", "game_year", "game_date", "game_pk")]
    out[cols] = pg[cols].to_numpy()[order]
    return out


def A5():
    pg = pitcher_panel()
    K = 10
    W = pitcher_windows(pg, 200, 200); W = W[W["season_n"] >= 1000]
    W0s = []
    for i in range(K):
        x = pitcher_windows(shuffle_pitchers(pg, 11 + i), 200, 200); W0s.append(x[x["season_n"] >= 1000])
    out = {"n_points": int(len(W)), "n_pitchers": int(W["pitcher"].nunique()), "n_shuffles": K, "persistence": {}, "cross_xwoba_ag": {}}

    def sl(D, m):
        D = D.dropna(subset=[f"rec_{m}", f"fwd_{m}", f"orc_{m}"])
        x, y = D[f"rec_{m}"] - D[f"orc_{m}"], D[f"fwd_{m}"] - D[f"orc_{m}"]
        f = C.ols_cl(y.to_numpy(), x.to_numpy(), D["pitcher"].to_numpy())
        return float(f.params[1]), float(f.bse[1]), int(len(D))
    PMa = [m for m in PM if f"rec_{m}" in W]
    for m in PMa:
        a = sl(W, m); nl = [sl(x, m) for x in W0s]
        b = (float(np.mean([v[0] for v in nl])), float(np.mean([v[1] for v in nl])))
        out["persistence"][m] = {"label": PM[m][3], "persistence": a[0] - b[0], "se": float(np.hypot(a[1], b[1])),
                                 "slope_actual": a[0], "slope_null": b[0], "n": a[2]}

    def cp(D, m):
        D = D.dropna(subset=[f"rec_{m}", f"orc_{m}", "fwd_xwoba_ag", "orc_xwoba_ag"])
        x = D[f"rec_{m}"] - D[f"orc_{m}"]
        z = x / x.std()
        y = (D["fwd_xwoba_ag"] - D["orc_xwoba_ag"]) * 1000
        f = C.ols_cl(y.to_numpy(), z.to_numpy(), D["pitcher"].to_numpy())
        return float(f.params[1]), float(f.bse[1])
    for m in PMa:
        a = cp(W, m); nl = [cp(x, m) for x in W0s]
        b = (float(np.mean([v[0] for v in nl])), float(np.mean([v[1] for v in nl])))
        out["cross_xwoba_ag"][m] = {"pts_per_sd": a[0] - b[0], "se": float(np.hypot(a[1], b[1])), "raw": a[0], "null": b[0]}
    print("A5 persistence", {m: round(v["persistence"], 3) for m, v in out["persistence"].items()}, flush=True)
    print("A5 cross", {m: round(v["pts_per_sd"], 2) for m, v in out["cross_xwoba_ag"].items()}, flush=True)

    # starters: last three starts (results vs velocity) -> next three starts
    st = pg[pg["pa"] >= 15].reset_index(drop=True)
    res = {}

    def starters(S_):
        Wst = pitcher_windows(S_, 3, 3, unit="games")
        Wst = Wst[Wst["season_n"] >= 1000].dropna(subset=["rec_woba_ag", "rec_ff_velo", "orc_ff_velo", "orc_woba_ag", "fwd_xwoba_ag", "orc_xwoba_ag"])
        zr = (Wst["rec_woba_ag"] - Wst["orc_woba_ag"]); zr = zr / zr.std()
        zv = (Wst["rec_ff_velo"] - Wst["orc_ff_velo"]); zv = zv / zv.std()
        y = (Wst["fwd_xwoba_ag"] - Wst["orc_xwoba_ag"]) * 1000
        yw = (Wst["fwd_woba_ag"] - Wst["orc_woba_ag"]) * 1000
        f = C.ols_cl(y.to_numpy(), np.column_stack([zr, zv]), Wst["pitcher"].to_numpy())
        fw = C.ols_cl(yw.to_numpy(), np.column_stack([zr, zv]), Wst["pitcher"].to_numpy())
        return {"n": int(len(Wst)), "results_pts_per_sd": [float(f.params[1]), float(f.bse[1])],
                "velo_pts_per_sd": [float(f.params[2]), float(f.bse[2])],
                "woba_outcome_results": [float(fw.params[1]), float(fw.bse[1])], "woba_outcome_velo": [float(fw.params[2]), float(fw.bse[2])],
                "sd_velo_dev_mph": float((Wst["rec_ff_velo"] - Wst["orc_ff_velo"]).std())}
    res["actual"] = starters(st)
    nulls = [starters(shuffle_pitchers(st, 40 + i)) for i in range(K)]
    res["null"] = {k: [float(np.mean([n[k][0] for n in nulls])), float(np.mean([n[k][1] for n in nulls]))]
                   for k in ["results_pts_per_sd", "velo_pts_per_sd", "woba_outcome_results", "woba_outcome_velo"]}
    res["results_effect"] = res["actual"]["results_pts_per_sd"][0] - res["null"]["results_pts_per_sd"][0]
    res["velo_effect"] = res["actual"]["velo_pts_per_sd"][0] - res["null"]["velo_pts_per_sd"][0]
    res["results_se"] = float(np.hypot(res["actual"]["results_pts_per_sd"][1], res["null"]["results_pts_per_sd"][1]))
    res["velo_se"] = float(np.hypot(res["actual"]["velo_pts_per_sd"][1], res["null"]["velo_pts_per_sd"][1]))
    out["starters_results_vs_velocity"] = res
    out["source_extra_columns"] = "Statcast_Shared full-width cache" if SHARED else None
    out["park_week_adjusted"] = bool(SHARED) and os.environ.get("PITCH_ADJUST", "1") == "1"
    print("A5 starters", res["results_effect"], res["velo_effect"], flush=True)
    dump("a5", out)


# =========================================================================================== A6
def A6():
    bg, lg = panel()
    M = marcel()
    W = C.windows(bg, 50, 50, lg)
    D, resid, base = forecast_frame(W)
    _, comp, coefs = C.form_summary(D, resid, C.SWING)
    Wall = C.prep(C.windows(bg, 50, 50, lg, at_risk=True), 200)
    last = Wall.sort_values("game_date").groupby(["batter", "game_year"]).tail(1)
    last = last[last["game_year"].isin([2024, 2025])].copy()
    last["form"] = [sum(coefs[int(y)][c] * r[c] for c in C.SWING) for y, (_, r) in zip(last["game_year"], last[C.SWING].iterrows())]
    ps = []
    for tok in open(os.path.join(PUB, "postseason_woba_2024_2025.txt")).read().split():
        b, y, num, den, pa = tok.split(":")
        ps.append({"batter": int(b), "game_year": 2020 + int(y), "post_woba": int(num) / 1000 / max(int(den), 1) * 1000,
                   "post_den": int(den), "post_pa": int(pa)})
    P = pd.DataFrame(ps).merge(last, on=["batter", "game_year"], how="inner").merge(M, on=["batter", "game_year"], how="left")
    P = P[P["post_pa"] >= 10].copy()
    P["marcel"] = P["marcel"].fillna(P["marcel"].median())
    P["std_w"] = P["std_woba"] * 1000
    sdf = float(comp.std())
    P["form_sd"] = P["form"] / sdf
    X = sm.add_constant(P[["marcel", "std_w", "form_sd"]])
    f = sm.WLS(P["post_woba"], X, weights=P["post_den"]).fit(cov_type="cluster", cov_kwds={"groups": P["batter"]})
    b, s = float(f.params["form_sd"]), float(f.bse["form_sd"])
    # what the regular-season calibration predicts per SD of form
    expect = float(sdf)  # calibration slope ~1 => one SD of the form index = sdf wOBA points
    out = {"n_hitter_postseasons": int(len(P)), "hitters": int(P["batter"].nunique()), "total_pa": int(P["post_pa"].sum()),
           "pts_per_sd_form": b, "se": s, "ci95": [b - 1.96 * s, b + 1.96 * s], "mde_80pct_power": 2.8 * s,
           "regular_season_pts_per_sd_form": expect, "form_sd_pts": sdf}
    print("A6", out, flush=True)
    dump("a6", out)


# =========================================================================================== A7
A7_ROWS = {}


def A7_core(rule):
    bg, lg = panel()
    W = C.windows(bg, 50, 50, lg)
    D, resid, base = forecast_frame(W)
    s, comp, coefs = C.form_summary(D, resid, C.SWING)
    D["form"] = comp
    X0 = sm.add_constant(D[C.STRONG_BASE]).copy()
    X0["fwd_same"] = 0.0; X0["fwd_oppq"] = 0.0; X0["fwd_park"] = 0.0
    D["talent"] = base.predict(X0)
    D["rec_w"] = D["rec_woba"] * 1000

    def lineup():
        d = C.raw(DATA)
        d = d[["game_pk", "game_date", "game_year", "at_bat_number", "batter", "pitcher", "stand", "p_throws", "inning_topbot",
               "home_team", "away_team", "woba_denom"]].copy()
        d["team"] = np.where(d["inning_topbot"] == "Top", d["away_team"], d["home_team"])
        pa = d.drop_duplicates(["game_pk", "at_bat_number"])
        first = pa.sort_values("at_bat_number").groupby(["game_pk", "team", "batter"], sort=False).head(1)
        first = first.sort_values(["game_pk", "team", "at_bat_number"])
        first["slot"] = first.groupby(["game_pk", "team"]).cumcount()
        starters = first[first["slot"] < 9][["game_pk", "game_date", "game_year", "team", "batter"]]
        sp = pa.sort_values("at_bat_number").groupby(["game_pk", "team"]).head(1)[["game_pk", "team", "pitcher", "p_throws"]]
        sp = sp.rename(columns={"pitcher": "sp", "p_throws": "sp_hand"})
        hand = pa.groupby("batter")["stand"].agg(lambda s: s.value_counts(normalize=True).to_dict())
        tpa = d[pd.to_numeric(d["woba_denom"], errors="coerce") == 1].groupby(["team", "game_year", "batter"]).size().rename("tpa").reset_index()
        # relief same-hand share by batter hand
        pa2 = pa.merge(sp, on=["game_pk", "team"])
        rel = pa2[pa2["pitcher"] != pa2["sp"]]
        q = {h: float((rel.loc[rel["stand"] == h, "p_throws"] == h).mean()) for h in ["L", "R"]}
        return starters, sp, hand, tpa, q
    starters, sp, hand, tpa, relq = cached("lineup", lineup)
    bh = {b: max(h, key=h.get) for b, h in hand.items() if max(h.values()) >= 0.9}
    st = starters.merge(sp, on=["game_pk", "team"])
    pairs, pair_T = [], {}
    games_all = st.drop_duplicates(["game_pk", "team"])[["game_pk", "team", "game_date", "game_year", "sp_hand"]]
    for (team, yr), g in st.groupby(["team", "game_year"]):
        tp = tpa[(tpa["team"] == team) & (tpa["game_year"] == yr)].set_index("batter")["tpa"]
        tg = games_all[(games_all["team"] == team) & (games_all["game_year"] == yr)].sort_values("game_date")
        cand = [b for b in g["batter"].unique() if tp.get(b, 0) >= 150 and b in bh]
        S = {b: g[g["batter"] == b] for b in cand}
        score = {}
        for L in [b for b in cand if bh[b] == "L"]:
            for R in [b for b in cand if bh[b] == "R"]:
                lo = max(S[L]["game_date"].min(), S[R]["game_date"].min())
                hi = min(S[L]["game_date"].max(), S[R]["game_date"].max())
                T = set(tg.loc[(tg["game_date"] >= lo) & (tg["game_date"] <= hi), "game_pk"])
                if len(T) < 40:
                    continue
                gl, gr = set(S[L]["game_pk"]) & T, set(S[R]["game_pk"]) & T
                if not gl or not gr or len(gl & gr) / len(gl | gr) > 0.15:
                    continue
                sc = len(gl ^ gr) / len(T)
                if sc < 0.6:
                    continue
                if (S[L].loc[S[L]["game_pk"].isin(T), "sp_hand"] == "R").mean() < 0.70 or \
                   (S[R].loc[S[R]["game_pk"].isin(T), "sp_hand"] == "L").mean() < 0.60:
                    continue
                score[(L, R)] = (sc, T)
        if rule == "plan":   # PLAN_2026-09-30 rule, applied to the full season (every qualifying combination)
            for L in [b for b in cand if bh[b] == "L"]:
                for R in [b for b in cand if bh[b] == "R"]:
                    gl, gr = set(S[L]["game_pk"]), set(S[R]["game_pk"])
                    if len(gl) < 20 or len(gr) < 20 or len(gl & gr) / len(gl | gr) > 0.15:
                        continue
                    if (S[L]["sp_hand"] == "R").mean() < 0.70 or (S[R]["sp_hand"] == "L").mean() < 0.60:
                        continue
                    pairs.append((team, yr, L, R))
                    pair_T[(team, yr, L, R)] = set(tg["game_pk"])
            continue
        # mutual best match: each hitter's most complementary opposite-handed partner
        bestL = {}
        bestR = {}
        for (L, R), (sc, T) in score.items():
            if sc > bestL.get(L, (0, None))[0]:
                bestL[L] = (sc, R)
            if sc > bestR.get(R, (0, None))[0]:
                bestR[R] = (sc, L)
        for L, (sc, R) in bestL.items():
            if bestR.get(R, (0, None))[1] == L:
                pairs.append((team, yr, L, R))
                pair_T[(team, yr, L, R)] = score[(L, R)][1]
    # platoon splits from PA before each decision (prior seasons + season to date), regressed to league
    a2 = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "results_adjusted.json")))
    lgs = {"L": a2["platoon"]["LHB_league_split_pts"] / 1000, "R": a2["platoon"]["RHB_league_split_pts"] / 1000}
    kreg = {"L": 1000, "R": 2200}
    bgs = bg.sort_values(["batter", "game_date"]).copy()
    bgs["opp_pa"] = bgs["pa"] - bgs["same_num"]
    bgs["woba_opp_num"] = bgs["woba_num"] - bgs["woba_same_num"]
    for c in ["same_num", "opp_pa", "woba_same_num", "woba_opp_num"]:
        bgs["c_" + c] = bgs.groupby("batter")[c].cumsum() - bgs[c]
    dec = []
    games = st.drop_duplicates(["game_pk", "team"])[["game_pk", "team", "game_date", "game_year", "sp_hand"]]
    Dk = D[["batter", "game_date", "talent", "form", "rec_w", "fwd_woba"]].copy()
    Dk["game_date"] = pd.to_datetime(Dk["game_date"])
    Dk["resid"] = resid.to_numpy()
    Dk = Dk.sort_values("game_date")
    bgs["game_date"] = pd.to_datetime(bgs["game_date"])
    for team, yr, L, R in pairs:
        gms = games[(games["team"] == team) & (games["game_year"] == yr)]
        s_ = st[(st["team"] == team) & (st["game_year"] == yr)]
        gl, gr = set(s_[s_["batter"] == L]["game_pk"]), set(s_[s_["batter"] == R]["game_pk"])
        gms = gms[gms["game_pk"].isin((gl ^ gr) & pair_T[(team, yr, L, R)])].sort_values("game_date")
        rows = {}
        for b in (L, R):
            st_b = pd.merge_asof(gms[["game_pk", "game_date", "sp_hand"]], Dk[Dk["batter"] == b].drop(columns="batter"), on="game_date",
                                 direction="backward", allow_exact_matches=True)
            sp_b = pd.merge_asof(gms[["game_pk", "game_date"]], bgs[bgs["batter"] == b][["game_date", "c_same_num", "c_opp_pa", "c_woba_same_num", "c_woba_opp_num"]],
                                 on="game_date", direction="backward", allow_exact_matches=True)
            h = bh[b]
            ns, no = sp_b["c_same_num"].fillna(0), sp_b["c_opp_pa"].fillna(0)
            obs = np.where((ns > 0) & (no > 0), sp_b["c_woba_opp_num"] / no.clip(lower=1) - sp_b["c_woba_same_num"] / ns.clip(lower=1), lgs[h])
            split = lgs[h] + (obs - lgs[h]) * ns / (ns + kreg[h])
            st_b["split"] = split * 1000
            st_b["p_same"] = 0.6 * (st_b["sp_hand"] == h) + 0.4 * relq[h]
            rows[b] = st_b
        Lr, Rr = rows[L], rows[R]
        ok = Lr["talent"].notna() & Rr["talent"].notna()
        for i in np.where(ok)[0]:
            l, r = Lr.iloc[i], Rr.iloc[i]
            e0 = {"L": l["talent"] - l["split"] * l["p_same"], "R": r["talent"] - r["split"] * r["p_same"]}
            e1 = {"L": e0["L"] + l["form"], "R": e0["R"] + r["form"]}
            a = "L" if l["sp_hand"] == "R" else "R"
            bchoice = "L" if l["rec_w"] >= r["rec_w"] else "R"
            c0 = max(e0, key=e0.get)
            c1 = max(e1, key=e1.get)
            # B3 (plan addendum D): the same talent-only call with the league split for each hand in place of the
            # hitter's own regressed split; reported, never used by (a)-(c)
            e0L = {"L": l["talent"] - lgs["L"] * 1000 * l["p_same"], "R": r["talent"] - lgs["R"] * 1000 * r["p_same"]}
            c0L = max(e0L, key=e0L.get)
            dec.append({"team": team, "yr": yr, "L": L, "R": R, "date": l["game_date"], "sp_hand": l["sp_hand"],
                        "a": a, "b": bchoice, "c0": c0, "c": c1,
                        "val_b": e1[bchoice] - e1[a], "val_c0": e1[c0] - e1[a], "val_c": e1[c1] - e1[a],
                        "form_gap": (l["form"] - r["form"]), "talent_gap": l["talent"] - r["talent"],
                        "split_gap": (l["split"] * l["p_same"] - r["split"] * r["p_same"]),
                        "resid_L": l["resid"], "resid_R": r["resid"],
                        "c0L": c0L, "val_c0L": e1[c0L] - e1[a], "split_L": l["split"], "split_R": r["split"],
                        "p_same_L": l["p_same"], "p_same_R": r["p_same"]})
    Dd = pd.DataFrame(dec)
    A7_ROWS[rule] = Dd   # decision rows kept for the B3 per-pair table (b3_a7_pairs.py)
    tor = lambda x: x / WOBA_SCALE * 50 / 1000   # wOBA points -> runs per 50 PA
    out = {"n_pairs": len(pairs), "n_decisions": int(len(Dd)), "relief_same_hand_share": relq,
           "n_team_seasons": len({(t, y) for t, y, _, _ in pairs}), "n_players": len({p for _, _, l, r in pairs for p in (l, r)}),
           "pairing_rule": "same team-season, opposite hands, >=150 PA each; over the overlap of their tenures (>=40 team games) exactly one starts in >=60% of games, co-start <=15%, LHB >=70% of starts vs RHP, RHB >=60% vs LHP; mutual best complementary partner",
           "pairs": [[t, int(y), int(l), int(r)] for t, y, l, r in pairs]}
    if len(Dd):
        for v in ["b", "c0", "c"]:
            x = tor(Dd[f"val_{v}"])
            per_pair = Dd.assign(x=x).groupby(["team", "yr", "L", "R"])["x"].mean()
            out[f"option_{v}_vs_strict_platoon_runs_per50"] = {"mean": float(x.mean()), "p10": float(x.quantile(.1)), "p50": float(x.quantile(.5)),
                                                                "p90": float(x.quantile(.9)), "share_differs_from_platoon": float((Dd[v] != Dd["a"]).mean()),
                                                                "per_pair_p10_p50_p90": [float(per_pair.quantile(q)) for q in (.1, .5, .9)]}
        out["share_form_flips_call"] = float((Dd["c"] != Dd["c0"]).mean())
        out["share_talent_split_overrides_platoon"] = float((Dd["c0"] != Dd["a"]).mean())
        fl = Dd[Dd["c"] != Dd["c0"]]
        if len(fl):
            # realised check: residual (next-50-PA wOBA minus strong baseline) of the form-favoured hitter minus the other
            fav = np.where(fl["c"] == "L", fl["resid_L"] - fl["resid_R"], fl["resid_R"] - fl["resid_L"])
            pred = np.where(fl["c"] == "L", fl["form_gap"], -fl["form_gap"])
            g = fl[["team", "yr", "L", "R"]].astype(str).agg("_".join, axis=1)
            m = sm.OLS(fav, np.ones(len(fav))).fit(cov_type="cluster", cov_kwds={"groups": g.to_numpy()})
            out["flipped_realised_minus_other_pts"] = [float(m.params[0]), float(m.bse[0])]
            out["flipped_predicted_form_gap_pts"] = float(np.mean(pred))
            out["n_flipped"] = int(len(fl))
        out["mean_abs_gaps_pts"] = {"form": float(Dd["form_gap"].abs().mean()), "talent": float(Dd["talent_gap"].abs().mean()),
                                    "split_term": float(Dd["split_gap"].abs().mean())}
    if rule == "plan":
        out["pairing_rule"] = "PLAN_2026-09-30: same team-season, opposite hands, >=150 PA each, >=20 starts each, co-start <=15%, LHB >=70% of starts vs RHP, RHB >=60% vs LHP (every qualifying combination)"
    print("A7", rule, {k: v for k, v in out.items() if k not in ("pairs", "pairing_rule")}, flush=True)
    return out


def A7():
    dump("a7", {"prespecified": A7_core("plan"), "refined_exploratory": A7_core("mutual"),
                "note": "The prespecified rule counts every qualifying left/right combination, so one hitter can sit in several pairs "
                        "(up to 13 per team-season). The refined rule (added after the first results; exploratory) keeps each hitter's "
                        "single most complementary partner over their shared tenure."})


# =========================================================================================== A8
def A8():
    bg, lg = panel()
    P, A = il_events()
    LABEL_END = pd.Timestamp("2026-09-27")
    HOR = 30
    W = C.windows(bg, 50, 50, lg)
    W0 = C.windows(C.shuffle_games(bg, SEED), 50, 50, lg)
    W = W[W["season_pa"] >= 300].copy()
    W["game_date"] = pd.to_datetime(W["game_date"])
    W = W[(W["game_date"] + pd.Timedelta(days=HOR) <= LABEL_END)]
    Pb = {b: g.sort_values("start") for b, g in P.groupby("batter")}
    Ab = {b: g.sort_values("end") for b, g in A.groupby("batter")}
    il_next, il_fam, ret = np.zeros(len(W), int), np.array([""] * len(W), dtype=object), np.zeros(len(W), int)
    for n, (dt, b) in enumerate(zip(W["game_date"], W["batter"])):
        g = Pb.get(b)
        if g is not None:
            h = g[(g["start"] >= dt) & (g["start"] <= dt + pd.Timedelta(days=HOR))]
            if len(h):
                il_next[n] = 1; il_fam[n] = h["family"].iloc[0]
        a = Ab.get(b)
        if a is not None and ((a["end"] <= dt) & (a["end"] >= dt - pd.Timedelta(days=45))).any():
            ret[n] = 1
    W["il_next"], W["il_fam"], W["recent_return"] = il_next, il_fam, ret
    ok = (W["rec_bat_speed__n"] >= 30) & (W["orc_bat_speed__n"] >= 90) & (W["fwd_bat_speed__n"] >= 30)
    D = W[ok].copy()
    D["bs_dev"] = D["rec_bat_speed"] - D["orc_bat_speed"]; D["bs_fwd"] = D["fwd_bat_speed"] - D["orc_bat_speed"]
    D["bs_dev_prior"] = D["rec_bat_speed"] - D["prior_bat_speed"]
    D["woba_dev_prior"] = D["rec_woba"] - D["prior_woba"]
    sd_bs = D["bs_dev_prior"].std()
    D["z_bs"] = D["bs_dev_prior"] / sd_bs
    D["z_w"] = D["woba_dev_prior"] / D["woba_dev_prior"].std()
    out = {"n_points": int(len(D)), "base_rate": float(D["il_next"].mean()), "share_recent_return": float(D["recent_return"].mean()),
           "n_placements_in_window": int(len(P)), "sd_bs_dev_mph": float(sd_bs), "risk": {}}
    base = D[D["recent_return"] == 0]
    for name, cols in [("bat_speed_only", ["z_bs"]), ("woba_only", ["z_w"]), ("both", ["z_bs", "z_w"])]:
        f = sm.Logit(base["il_next"], sm.add_constant(base[cols])).fit(disp=0, cov_type="cluster", cov_kwds={"groups": base["batter"]})
        out["risk"][name] = {c: {"OR_per_minus1SD": float(np.exp(-f.params[c])),
                                 "ci": [float(np.exp(-f.params[c] - 1.96 * f.bse[c])), float(np.exp(-f.params[c] + 1.96 * f.bse[c]))]} for c in cols}
    for fam in ["oblique", "hamstring", "hand_wrist"]:
        y = (base["il_fam"] == fam).astype(int)
        f = sm.Logit(y, sm.add_constant(base[["z_bs"]])).fit(disp=0, cov_type="cluster", cov_kwds={"groups": base["batter"]})
        out["risk"][fam] = {"OR_per_minus1SD": float(np.exp(-f.params["z_bs"])),
                            "ci": [float(np.exp(-f.params["z_bs"] - 1.96 * f.bse["z_bs"])), float(np.exp(-f.params["z_bs"] + 1.96 * f.bse["z_bs"]))],
                            "n_events": int(y.sum())}
    q = pd.qcut(base["z_bs"].rank(method="first"), 10, labels=False)
    dec = base.groupby(q).agg(il_rate=("il_next", "mean"), n=("il_next", "size"), dev_mph=("bs_dev_prior", "mean")).reset_index(drop=True)
    out["il_rate_by_decile"] = dec.to_dict(orient="list")
    Xq = sm.add_constant(pd.DataFrame({"z": base["z_bs"], "z2": base["z_bs"] ** 2}))
    fq = sm.Logit(base["il_next"], Xq).fit(disp=0, cov_type="cluster", cov_kwds={"groups": base["batter"]})
    out["quadratic_prespecified_secondary"] = {"b_z": [float(fq.params["z"]), float(fq.bse["z"])], "b_z2": [float(fq.params["z2"]), float(fq.bse["z2"])],
                                               "p_z2": float(fq.pvalues["z2"]), "vertex_z": float(-fq.params["z"] / (2 * fq.params["z2"])),
                                               "prediction": "b_z2 < 0 with vertex > 0",
                                               "supported": bool(fq.params["z2"] < 0 and fq.pvalues["z2"] < 0.05 and -fq.params["z"] / (2 * fq.params["z2"]) > 0)}
    ok0 = (W0["season_pa"] >= 300) & (W0["rec_bat_speed__n"] >= 30) & (W0["orc_bat_speed__n"] >= 90) & (W0["fwd_bat_speed__n"] >= 30)
    N0 = W0[ok0]
    null = float(C.ols_cl((N0["fwd_bat_speed"] - N0["orc_bat_speed"]).to_numpy(), (N0["rec_bat_speed"] - N0["orc_bat_speed"]).to_numpy(), N0["batter"].to_numpy()).params[1])
    groups = {"all": D, "no_IL_next30_no_recent_return": D[(D["il_next"] == 0) & (D["recent_return"] == 0)],
              "IL_next30": D[D["il_next"] == 1], "IL_next30_three_families": D[D["il_fam"].isin(["oblique", "hamstring", "hand_wrist"])],
              "recent_return_45d": D[D["recent_return"] == 1]}
    out["null_slope"] = null
    out["persistence_by_group"] = {}
    for name, G in groups.items():
        f = C.ols_cl(G["bs_fwd"].to_numpy(), G["bs_dev"].to_numpy(), G["batter"].to_numpy())
        out["persistence_by_group"][name] = {"n": int(len(G)), "persistence": float(f.params[1] - null), "se": float(f.bse[1])}
    cold = D[D["z_bs"] <= -1]
    out["cold_tail"] = {"n": int(len(cold)), "share_IL_next30": float(cold["il_next"].mean()), "base": float(D["il_next"].mean())}
    print("A8", json.dumps({k: v for k, v in out.items() if k != "il_rate_by_decile"}, default=str)[:1500], flush=True)
    dump("a8", out)


# =========================================================================================== A9
def runs_z(x):
    x = np.asarray(x, int)
    n1 = x.sum(); n2 = len(x) - n1; n = len(x)
    if n1 == 0 or n2 == 0:
        return np.nan
    R = 1 + np.sum(x[1:] != x[:-1])
    E = 2 * n1 * n2 / n + 1
    V = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n * n * (n - 1))
    return (R - E) / np.sqrt(V)


def A9():
    out = {}
    d = C.raw(DATA)
    d = d.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
    pa = d[pd.to_numeric(d["woba_denom"], errors="coerce") == 1][["batter", "game_year", "game_date", "game_pk", "at_bat_number", "woba_value"]].copy()
    pa["onb"] = (pd.to_numeric(pa["woba_value"], errors="coerce").fillna(0) > 0).astype(int)
    dd = C.pitches(DATA)
    sw = dd[dd["comp"] == 1][["batter", "game_year", "game_date", "game_pk", "at_bat_number", "pitch_number", "bat_speed"]].copy()
    sw = sw.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
    sw["fast"] = (sw["bat_speed"] >= sw.groupby(["batter", "game_year"])["bat_speed"].transform("median")).astype(int)
    rng = np.random.default_rng(SEED)

    def by_season(df, col, min_n):
        zs, z0s = [], []
        for (b, y), g in df.groupby(["batter", "game_year"], sort=False):
            if len(g) < min_n:
                continue
            zs.append(runs_z(g[col].to_numpy()))
            gm = g["game_pk"].unique()
            perm = dict(zip(gm, rng.permutation(len(gm))))
            g0 = g.assign(o=g["game_pk"].map(perm)).sort_values(["o"], kind="stable")
            z0s.append(runs_z(g0[col].to_numpy()))
        z, z0 = np.array(zs), np.array(z0s)
        dlt = z - z0
        return {"n_hitter_seasons": int(len(z)), "mean_z": float(np.nanmean(z)), "sd_z": float(np.nanstd(z)),
                "share_z_below_-1.96": float(np.nanmean(z < -1.96)), "mean_z_game_shuffle": float(np.nanmean(z0)),
                "sd_z_game_shuffle": float(np.nanstd(z0)), "share_below_-1.96_shuffle": float(np.nanmean(z0 < -1.96)),
                "mean_diff": float(np.nanmean(dlt)), "se_diff": float(np.nanstd(dlt) / np.sqrt(np.sum(~np.isnan(dlt))))}
    out["albright_runs"] = {"on_base_by_PA": by_season(pa, "onb", 300), "bat_speed_above_median_by_swing": by_season(sw, "fast", 600)}
    print("A9 runs", out["albright_runs"], flush=True)

    # Green & Zwiebel: recent 25 PA vs season rate excluding +-50 PA around the decision; top/bottom 5%
    bg, lg = panel()
    bg0 = C.shuffle_games(bg, SEED)
    gz = {}
    for tag, b in [("actual", bg), ("null", bg0)]:
        W25 = C.windows(b, 25, 25, lg, extra=("onb", "hr"))
        W50 = C.windows(b, 50, 50, lg, extra=("onb", "hr"))
        J = W25.merge(W50[["row"] + [c for c in W50.columns if c.startswith("orc_")]], on="row", suffixes=("", "_50"))
        J = J[J["season_pa"] >= 300]
        res = {}
        for hot_m, rate_col_rec, rate_col_orc in [("woba", "rec_woba", "orc_woba_50"), ("onb", "rec_onb", "orc_onb_50"),
                                                  ("hr", "rec_hr", "orc_hr_50"), ("bat_speed", "rec_bat_speed", "orc_bat_speed_50")]:
            Jm = J.dropna(subset=[rate_col_rec, rate_col_orc])
            if hot_m == "bat_speed":
                Jm = Jm[(Jm["rec_bat_speed__n"] >= 20) & (Jm["orc_bat_speed__n_50"] >= 60)]
            dev = Jm[rate_col_rec] - Jm[rate_col_orc]
            hi, lo = dev.quantile(0.95), dev.quantile(0.05)
            hot, cold = (dev >= hi).astype(float), (dev <= lo).astype(float)
            r2 = {}
            for outc, fc, oc, scale in [("woba", "fwd_woba", "orc_woba_50", 1000), ("onb", "fwd_onb", "orc_onb_50", 1000), ("hr", "fwd_hr", "orc_hr_50", 100)]:
                y = (Jm[fc] - Jm[oc]) * scale
                f = C.ols_cl(y.to_numpy(), np.column_stack([hot, cold]), Jm["batter"].to_numpy())
                r2[outc] = {"hot": [float(f.params[1]), float(f.bse[1])], "cold": [float(f.params[2]), float(f.bse[2])],
                            "base_rate": float(Jm[oc].mean() * scale)}
            res[hot_m] = r2
        gz[tag] = res
    corr = {}
    for hm in gz["actual"]:
        corr[hm] = {}
        for oc in gz["actual"][hm]:
            a, n = gz["actual"][hm][oc], gz["null"][hm][oc]
            corr[hm][oc] = {"hot": a["hot"][0] - n["hot"][0], "hot_se": float(np.hypot(a["hot"][1], n["hot"][1])),
                            "cold": a["cold"][0] - n["cold"][0], "cold_se": float(np.hypot(a["cold"][1], n["cold"][1])),
                            "hot_raw": a["hot"][0], "cold_raw": a["cold"][0], "base_rate": a["base_rate"]}
    out["green_zwiebel"] = {"raw_and_null": gz, "null_corrected": corr,
                            "their_reported": {"hr": "hot-in-HR batter 15-25% more likely to homer next AB (0.5-0.75 pp)",
                                               "obp": "about 25 OBP points for a 0.4 gap in recent OBP",
                                               "summary": "hot = one-half to one SD of the distribution of player ability"}}
    print("A9 GZ", json.dumps(corr)[:1500], flush=True)
    dump("a9", out)


# =========================================================================================== A10
def A10():
    out = {}
    bg, lg = panel()
    bg0 = C.shuffle_games(bg, SEED)
    sb = season_bounds(bg)

    def run(tag, b, b0, lg_, forecast=True, **kw):
        W = C.windows(b, lg=lg_, **kw)
        W0 = [C.windows(C.shuffle_games(b, SEED + i), lg=lg_, **kw) for i in range(5)]
        r = {"persistence": C.persistence(W, W0, HEAD)}
        if forecast:
            D, resid, _ = forecast_frame(W)
            s, _, _ = C.form_summary(D, resid, C.SWING)
            s2, _, _ = C.form_summary(D, resid, ["dev_woba"])
            r["swing_form"] = {k: s[k] for k in ["top", "top_se", "bottom", "bottom_se", "oos_r2_gain", "n"]}
            r["results_form"] = {k: s2[k] for k in ["top", "top_se", "bottom", "bottom_se", "oos_r2_gain"]}
        out[tag] = r
        print("A10", tag, {m: round(v["persistence"], 3) for m, v in r["persistence"].items()}, r.get("swing_form", {}).get("top"), r.get("swing_form", {}).get("bottom"), flush=True)
        dump("a10", out)

    run("primary_50_50", bg, bg0, lg, rec=50, fwd=50)
    run("w100_100", bg, bg0, lg, rec=100, fwd=100)
    run("days14_14", bg, bg0, lg, rec=14, fwd=14, mode="days", season_bounds=sb)
    run("days28_28", bg, bg0, lg, rec=28, fwd=28, mode="days", season_bounds=sb)
    W = C.windows(bg, 50, 50, lg); W0 = [C.windows(C.shuffle_games(bg, SEED + i), 50, 50, lg) for i in range(5)]
    D, resid, _ = forecast_frame(W)
    s, comp, _ = C.form_summary(D, resid, C.SWING)
    Dy = D.assign(form=comp)
    for yr in [2024, 2025, 2026]:
        out[f"season_{yr}"] = {"persistence": C.persistence(W[W["game_year"] == yr], [x[x["game_year"] == yr] for x in W0], HEAD)}
        m = Dy["game_year"] == yr
        e = C.decile_effects(Dy[m], resid[m], "form")
        out[f"season_{yr}"]["swing_form"] = {"top": e["bump"][9], "top_se": e["se"][9], "bottom": e["bump"][0], "bottom_se": e["se"][0]}
        print("A10 season", yr, out[f"season_{yr}"]["swing_form"], flush=True)
    # first 30 days of each season removed
    bgf = bg.copy()
    first = bgf.groupby("game_year")["game_date"].transform("min")
    bgf = bgf[bgf["game_date"] >= first + pd.Timedelta(days=30)].reset_index(drop=True)
    run("drop_first_30_days", bgf, C.shuffle_games(bgf, SEED), lg, forecast=False, rec=50, fwd=50)
    # competitive-swing filter off; park/week adjustment off
    bgn, lgn = panel(adjusted=True, comp=False)
    run("competitive_filter_off", bgn, C.shuffle_games(bgn, SEED), lgn, rec=50, fwd=50)
    bgu, lgu = panel(adjusted=False, comp=True)
    run("park_week_adjustment_off", bgu, C.shuffle_games(bgu, SEED), lgu, rec=50, fwd=50)
    dump("a10", out)


if __name__ == "__main__":
    for k in range(1, 11):
        name = f"A{k}"
        if name in WANT:
            t = time.time()
            print(f"==== {name}", flush=True)
            globals()[name]()
            print(f"==== {name} done in {time.time() - t:.0f}s", flush=True)
