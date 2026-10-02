"""Shared machinery for analysis3.py (A1-A10). Reuses hot_streak_study.py definitions; the window builder is
generalised (calendar windows, cross-season forward windows, at-risk rows) and the analysis2 forecast frame is
reproduced function-for-function so the primary numbers are unchanged."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import hot_streak_study as H

SUM_COLS = H.SUM_COLS
METRICS = H.METRICS
ALLM = list(H.METRICS)
STRONG_BASE = ([f"prior_{m}" for m in ALLM] + ["std_woba", "std_xwoba", "std_bat_speed", "std_ev", "std_k_rate",
               "std_chase", "std_whiff"] + ["fwd_oppq", "fwd_park", "fwd_same"])
SWING = ["dev_bat_speed", "dev_fast_swing", "dev_swing_len"]


# ------------------------------------------------------------------ data
_RAW = {}


def raw(data_dir):
    if data_dir not in _RAW:
        _RAW[data_dir] = H.load(data_dir)
    return _RAW[data_dir]


def adjust(d):
    """analysis2.py park-season and week-of-season adjustment (identical code)."""
    d = d.copy()
    wk = d["game_date"].dt.isocalendar().week.astype(int)
    for c in ["bat_speed", "swing_length", "launch_speed"]:
        x = pd.to_numeric(d[c], errors="coerce")
        ymean = x.groupby(d["game_year"]).transform("mean")
        park = x.groupby([d["home_team"], d["game_year"]]).transform("mean") - ymean
        week = x.groupby([d["game_year"], wk]).transform("mean") - ymean
        d[c] = x - park - week
    return d


def no_comp_filter(d):
    """Every tracked swing counts (competitive-swing filter off)."""
    bs = pd.to_numeric(d["bat_speed"], errors="coerce")
    sl = pd.to_numeric(d["swing_length"], errors="coerce")
    ev = pd.to_numeric(d["launch_speed"], errors="coerce")
    comp = bs.notna() & sl.notna()
    d["comp"] = comp.astype(np.int8)
    d["bs_num"] = np.where(comp, bs, 0.0)
    d["sl_num"] = np.where(comp, sl, 0.0)
    d["fast"] = (comp & (bs >= 75)).astype(np.int8)
    plate_speed = pd.to_numeric(d["release_speed"], errors="coerce") * 0.92
    ratio = ev / (1.23 * bs + 0.23 * plate_speed)
    contact = comp & (d["type"] == "X") & ev.notna()
    d["sq_up"] = (contact & (ratio >= 0.80)).astype(np.int8)
    d["blast"] = (contact & (ratio * 100 + bs >= 164)).astype(np.int8)
    return d


def pitches(data_dir, adjusted=True, comp=True):
    d = raw(data_dir)
    d = adjust(d) if adjusted else d.copy()
    d = H.pitch_features(d)
    if not comp:
        d = no_comp_filter(d)
    d = H.context(d)
    return d


# ------------------------------------------------------------------ windows
def shuffle_games(bg, seed):
    """Permute each hitter-season's game contents over his own game slots (dates and ids stay in place)."""
    rng = np.random.default_rng(seed)
    bg = bg.sort_values(["batter", "game_date", "game_pk"]).reset_index(drop=True)
    key = rng.random(len(bg))
    order = pd.DataFrame({"b": bg["batter"], "y": bg["game_year"], "k": key}).sort_values(["b", "y", "k"]).index.to_numpy()
    out = bg.copy()
    cols = [c for c in bg.columns if c not in ("batter", "game_year", "game_date", "game_pk")]
    out[cols] = bg[cols].to_numpy()[order]
    return out


def windows(bg, rec, fwd, lg, mode="pa", cross_season=False, at_risk=False, season_bounds=None, extra=()):
    """Generalised H.build_windows. mode 'pa' (rec/fwd in PA) or 'days' (rec/fwd in calendar days).
    cross_season: forward window may run into the hitter's next season. at_risk: keep rows whose forward
    window is incomplete (flag 'complete'). Returns the same columns as H.build_windows plus 'row' and 'complete'."""
    bg = bg.sort_values(["batter", "game_date", "game_pk"]).reset_index(drop=True)
    ALL = list(SUM_COLS) + list(extra)
    X = bg[ALL].to_numpy(float)
    C = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, axis=0)])
    col = {c: i for i, c in enumerate(ALL)}
    cpa = C[:, col["pa"]]
    n = len(bg)
    idx = np.arange(n)
    key = bg["batter"].astype(str) + "_" + bg["game_year"].astype(str)
    s_start = idx - bg.groupby(key, sort=False).cumcount().to_numpy()
    s_end = s_start + bg.groupby(key, sort=False)["pa"].transform("size").to_numpy()
    b_start = idx - bg.groupby("batter", sort=False).cumcount().to_numpy()
    b_end = b_start + bg.groupby("batter", sort=False)["pa"].transform("size").to_numpy()
    if mode == "pa":
        j = np.searchsorted(cpa, cpa[idx] - rec, side="right") - 1
        k = np.searchsorted(cpa, cpa[idx] + fwd, side="left")
        ok = j >= s_start
        hi = b_end if cross_season else s_end
        complete = k <= hi
    else:
        dn = (bg["game_date"].to_numpy().astype("datetime64[D]").astype(np.int64)).astype(float)
        # games are sorted by batter then date; searchsorted within the batter block via an offset trick
        off = bg["batter"].to_numpy().astype(float) * 1e6
        t = off + dn
        j = np.searchsorted(t, t - rec, side="left")          # first game with date >= d - rec
        k = np.searchsorted(t, t + fwd, side="left")          # first game with date >= d + fwd
        j = np.maximum(j, s_start)
        k = np.minimum(k, s_end)
        yr = bg["game_year"].to_numpy()
        lo = np.array([season_bounds[y][0] for y in yr])
        hi_d = np.array([season_bounds[y][1] for y in yr])
        ok = (dn - rec >= lo)
        complete = (dn + fwd <= hi_d)
    if not at_risk:
        ok = ok & complete
    k = np.minimum(k, b_end if cross_season else s_end)
    i, j, k, comp = idx[ok], j[ok], k[ok], complete[ok]
    ss, se, bs = s_start[ok], s_end[ok], b_start[ok]

    def S(a, b):
        return {c: C[b, col[c]] - C[a, col[c]] for c in ALL}

    rec_, fwd_ = S(j, i), S(i, k)
    prior = S(bs, j)
    season = S(ss, se)
    fwd_in = S(i, np.minimum(k, se))
    oracle = {c: season[c] - rec_[c] - fwd_in[c] for c in ALL}
    std = S(ss, j)
    with np.errstate(invalid="ignore", divide="ignore"):
        cols = {
            "row": i, "batter": bg["batter"].to_numpy()[i], "game_year": bg["game_year"].to_numpy()[i],
            "game_date": bg["game_date"].to_numpy()[i], "game_pk": bg["game_pk"].to_numpy()[i],
            "season_pa": season["pa"], "prior_pa": prior["pa"], "fwd_pa": fwd_["pa"], "rec_pa": rec_["pa"],
            "fwd_oppq": fwd_["opp_q_num"] / fwd_["pa"], "fwd_park": fwd_["park_num"] / fwd_["pa"],
            "fwd_same": fwd_["same_num"] / fwd_["pa"], "complete": comp,
            "crosses": (k > se).astype(int),
        }
    for tag, Sx, reg in [("rec", rec_, False), ("fwd", fwd_, False), ("prior", prior, True),
                         ("orc", oracle, False), ("std", std, True)]:
        r = H.rates(Sx, lg, regress=reg)
        for m in METRICS:
            cols[f"{tag}_{m}"] = r[m]
            cols[f"{tag}_{m}__n"] = r[m + "__n"]
        for x in extra:
            with np.errstate(invalid="ignore", divide="ignore"):
                cols[f"{tag}_{x}"] = np.where(Sx["pa"] > 0, Sx[x] / Sx["pa"], np.nan)
    return pd.DataFrame(cols)


def ols_cl(y, x, g):
    X = sm.add_constant(np.asarray(x, float).reshape(len(y), -1), has_constant="add")
    return sm.OLS(np.asarray(y, float), X, missing="drop").fit(cov_type="cluster", cov_kwds={"groups": np.asarray(g)})


def slope(D, m, min_season_pa=300):
    dn = METRICS[m][1]
    mn = H.MIN_N[dn]
    ok = (D["season_pa"] >= min_season_pa) & (D[f"rec_{m}__n"] >= mn) & (D[f"fwd_{m}__n"] >= mn) & (D[f"orc_{m}__n"] >= 3 * mn)
    D = D[ok]
    x = D[f"rec_{m}"] - D[f"orc_{m}"]
    y = D[f"fwd_{m}"] - D[f"orc_{m}"]
    f = ols_cl(y.to_numpy(), x.to_numpy(), D["batter"].to_numpy())
    return float(f.params[1]), float(f.bse[1]), int(len(D))


def persistence(W, W0, metrics, min_season_pa=300):
    """W0 may be one shuffled frame or a list of them (null slope = mean over shuffles; SE uses the mean null SE,
    which is conservative relative to the Monte Carlo error of the averaged null)."""
    W0s = W0 if isinstance(W0, (list, tuple)) else [W0]
    out = {}
    for m in metrics:
        a, ase, n = slope(W, m, min_season_pa)
        nl = [slope(x, m, min_season_pa) for x in W0s]
        b, bse = float(np.mean([v[0] for v in nl])), float(np.mean([v[1] for v in nl]))
        out[m] = {"persistence": a - b, "se": float(np.hypot(ase, bse)), "slope_actual": a, "slope_null": b, "n": n,
                  "n_shuffles": len(W0s), "null_sd_across_shuffles": float(np.std([v[0] for v in nl], ddof=1)) if len(nl) > 1 else None}
    return out


# ------------------------------------------------------------------ forecast frame (analysis2, verbatim logic)
def prep(W, min_prior=200):
    D = W[(W["prior_pa"] >= min_prior)].copy()
    for m in ALLM:
        _, dn, kreg, _ = METRICS[m]
        n = D[f"rec_{m}__n"]
        D[f"dev_{m}"] = ((D[f"rec_{m}"] - D[f"prior_{m}"]) * n / (n + kreg)).fillna(0.0)
        D[f"raw_{m}"] = D[f"rec_{m}"] - D[f"prior_{m}"]
    for c in STRONG_BASE:
        D[c] = D[c].fillna(D[c].mean())
    return D


def resid_on(D, target, base_cols, weights=None):
    y = D[f"fwd_{target}"] * 1000
    w = D["fwd_pa"] if weights is None else D["fwd_pa"] * weights
    f = sm.WLS(y, sm.add_constant(D[base_cols]), weights=w).fit()
    return y - f.predict(sm.add_constant(D[base_cols])), f


def decile_effects(D, resid, var=None, q=None, weights=None):
    if q is None:
        ok = D[var].notna()
        D, resid = D[ok], resid[ok]
        q = pd.qcut(D[var].rank(method="first"), 10, labels=False)
    else:
        q = q.loc[D.index]
    X = pd.get_dummies(q, prefix="d", dtype=float)
    X = X.drop(columns=[c for c in ["d_4", "d_5"] if c in X])
    w = D["fwd_pa"] if weights is None else D["fwd_pa"] * weights.loc[D.index]
    fit = sm.WLS(resid, sm.add_constant(X), weights=w).fit(cov_type="cluster", cov_kwds={"groups": D["batter"]})
    out = {"decile": list(range(10)), "bump": [], "se": [], "n": [int((q == k).sum()) for k in range(10)]}
    for k in range(10):
        if k in (4, 5) or f"d_{k}" not in fit.params:
            out["bump"].append(0.0); out["se"].append(0.0)
        else:
            out["bump"].append(float(fit.params[f"d_{k}"])); out["se"].append(float(fit.bse[f"d_{k}"]))
    return out


def crossfit(D, resid, cols):
    comp = pd.Series(np.nan, index=D.index)
    coefs = {}
    for yr in sorted(D["game_year"].unique()):
        tr, te = D["game_year"] != yr, D["game_year"] == yr
        f = sm.WLS(resid[tr], sm.add_constant(D.loc[tr, cols]), weights=D.loc[tr, "fwd_pa"]).fit()
        comp[te] = f.predict(sm.add_constant(D.loc[te, cols], has_constant="add")) - f.params["const"]
        coefs[int(yr)] = {c: float(f.params[c]) for c in cols}
    return comp, coefs


def form_summary(D, resid, cols, weights=None, q=None):
    comp, coefs = crossfit(D, resid, cols)
    D = D.assign(form=comp)
    eff = decile_effects(D, resid, "form", q=q, weights=weights)
    mse0 = np.average(resid ** 2, weights=D["fwd_pa"])
    mse1 = np.average((resid - comp) ** 2, weights=D["fwd_pa"])
    return {"top": eff["bump"][9], "top_se": eff["se"][9], "bottom": eff["bump"][0], "bottom_se": eff["se"][0],
            "deciles": eff, "oos_r2_gain": float(1 - mse1 / mse0), "n": int(len(D)),
            "hitters": int(D["batter"].nunique())}, comp, coefs
