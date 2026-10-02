"""A11 (offseason analysis; plan: PLAN_ADDENDUM_2026-10-01c.md, written and hashed before this script was run).
Is the bat-speed "state" slow-reverting or a lasting shift?

python3 analysis4_reversion.py DATA_DIR PUBLIC_DIR OUT_DIR [real] [sims] [N_REPLICATES]

  real  : A11a/b/c on the real panel (+ survivorship, bootstrap) -> OUT_DIR/a11_real.json
  sims  : A11d calibration simulations on the real skeleton    -> OUT_DIR/a11_sims.json
  both present -> OUT_DIR/a11.json (merged, with the signature table and the decision rule applied)
FG_BAT (env) = FanGraphs position-player lines, for age where the Marcel file lacks it.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import least_squares

DATA, PUB, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PARTS = set(a for a in sys.argv[4:] if not a.isdigit())
NREP = int(next((a for a in sys.argv[4:] if a.isdigit()), 4))
sys.argv = [sys.argv[0], DATA, PUB, OUT]          # analysis3 parses argv at import
import analysis3 as A3                              # noqa: E402  (panel(), marcel(), fg_projection(), cache)
import common3 as C                                 # noqa: E402
import simulate as SIM                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = os.path.join(HERE, "..", "PLAN_ADDENDUM_2026-10-01c.md")
sha16 = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
SEEDS_SHUF = list(range(1101, 1121))
SEED_BOOT, SEED_SIM = 1131, 1141
TARGET_P = 0.26


def dump(name, obj):
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"wrote {name}.json", flush=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------ helpers
def ages():
    """age in season Y: Marcel file, FanGraphs Age where missing, median otherwise."""
    M = A3.marcel()[["batter", "game_year", "age", "marcel", "pa_prev"]]
    raw = []
    for tok in open(os.path.join(PUB, "marcel_woba_2024_2026.txt")).read().split():
        b, y, w, r, pa, age = tok.split(":")
        raw.append({"batter": int(b), "game_year": 2020 + int(y), "age_raw": float(age) if age else np.nan})
    M = M.merge(pd.DataFrame(raw), on=["batter", "game_year"], how="left")
    F = A3.fg_projection()
    if F is not None:
        M = M.merge(F[["batter", "game_year", "fg_age"]], on=["batter", "game_year"], how="outer")
        M["age_raw"] = M["age_raw"].fillna(M["fg_age"])
    M["age_y"] = M["age_raw"]
    return M[["batter", "game_year", "age_y", "marcel"]].drop_duplicates(["batter", "game_year"])


def ols(y, X, g, names):
    X = sm.add_constant(np.column_stack(X), has_constant="add")
    f = sm.OLS(np.asarray(y, float), X).fit(cov_type="cluster", cov_kwds={"groups": np.asarray(g)})
    return {n: (float(f.params[i + 1]), float(f.bse[i + 1])) for i, n in enumerate(names)}, int(len(y))


def wls(y, X, w, g, names):
    X = sm.add_constant(np.column_stack(X), has_constant="add")
    f = sm.WLS(np.asarray(y, float), X, weights=np.asarray(w, float)).fit(cov_type="cluster", cov_kwds={"groups": np.asarray(g)})
    return {n: (float(f.params[i + 1]), float(f.bse[i + 1])) for i, n in enumerate(names)}, int(len(y))


def changepoint(bs, comp):
    """A1 changepoint_adjust fit, verbatim algorithm: games with >=5 competitive swings, least squares, >=20 games each
    side, >=40 games. Returns (pre, post, k_game) with k_game = position (in the input arrays) of the first post game."""
    q = np.flatnonzero(comp >= 5)
    if len(q) < 40:
        return None
    y = bs[q] / comp[q]
    w = comp[q].astype(float)
    cw, cy, cyy = np.cumsum(w), np.cumsum(w * y), np.cumsum(w * y * y)
    ks = np.arange(20, len(q) - 20 + 1)
    w1, s1, ss1 = cw[ks - 1], cy[ks - 1], cyy[ks - 1]
    w2, s2, ss2 = cw[-1] - w1, cy[-1] - s1, cyy[-1] - ss1
    sse = (ss1 - s1 * s1 / w1) + (ss2 - s2 * s2 / w2)
    bk = int(ks[np.argmin(sse)])
    m1 = cy[bk - 1] / cw[bk - 1]
    m2 = (cy[-1] - cy[bk - 1]) / (cw[-1] - cw[bk - 1])
    return m1, m2, int(q[bk])


def hs_table(bg):
    """dict (batter, year) -> arrays in game order."""
    bg = bg.sort_values(["batter", "game_date", "game_pk"]).reset_index(drop=True)
    T = {}
    pa, comp, bs = bg["pa"].to_numpy().astype(float), bg["comp"].to_numpy().astype(float), bg["bs_num"].to_numpy().astype(float)
    dates = bg["game_date"].to_numpy()
    for key, idx in bg.groupby(["batter", "game_year"], sort=False).indices.items():
        idx = np.asarray(idx)
        T[(int(key[0]), int(key[1]))] = {"pa": pa[idx], "comp": comp[idx], "bs": bs[idx], "date": dates[idx]}
    return T


def mean_bs(h, sel):
    c = h["comp"][sel].sum()
    return h["bs"][sel].sum() / c if c > 0 else np.nan


def last_n(h, n):
    """boolean mask of the last games of the season reaching >= n PA (whole games)."""
    cs = np.cumsum(h["pa"][::-1])
    k = int(np.searchsorted(cs, n) + 1)
    m = np.zeros(len(h["pa"]), bool); m[len(m) - k:] = True
    return m, cs[-1] >= n


def first_n(h, n, start=0):
    cs = np.cumsum(h["pa"][start:])
    k = int(np.searchsorted(cs, n) + 1)
    m = np.zeros(len(h["pa"]), bool); m[start:start + k] = True
    return m, (len(cs) > 0 and cs[-1] >= n)


def blocks_of(h, start=0):
    """non-overlapping 50-PA blocks from game `start` (block = floor(PA before game / 50)); final block dropped if <50 PA;
    returns list of (t, mean, comp, first_game_pos) for blocks with >=30 competitive swings, and n complete blocks."""
    pa = h["pa"][start:]
    cpa = np.cumsum(pa) - pa
    blk = (cpa // 50).astype(int)
    out, ncomplete = [], 0
    for t in np.unique(blk):
        m = np.flatnonzero(blk == t)
        if pa[m].sum() < 50:
            continue
        ncomplete = t + 1
        c = h["comp"][start + m].sum()
        if c >= 30:
            out.append((int(t), h["bs"][start + m].sum() / c, c, start + int(m[0])))
    return out, ncomplete


# ------------------------------------------------------------------------------------------------ A11b
def a11b_frame(T, Tshuf=None):
    """one row per (hitter, Y) with >=300 PA in Y; Y in 2024, 2025. Tshuf: table whose season-Y game order is shuffled."""
    rows = []
    for (b, y), h in T.items():
        if y == 2026 or h["pa"].sum() < 300:
            continue
        hy = Tshuf[(b, y)] if Tshuf is not None else h
        nxt = T.get((b, y + 1))
        r = {"batter": b, "Y": y, "pa_Y": hy["pa"].sum(), "pa_next": 0.0 if nxt is None else nxt["pa"].sum()}
        L, _ = last_n(hy, 150)
        r["x"] = mean_bs(hy, L) - mean_bs(hy, ~L)
        rest_late = mean_bs(hy, ~L)
        # mid-season window: starts at the first game where PA before it >= 42.5% of season PA
        cpa = np.cumsum(hy["pa"]) - hy["pa"]
        s0 = int(np.searchsorted(cpa, 0.425 * hy["pa"].sum()))
        Mw, okm = first_n(hy, 150, s0)
        rest_mid = mean_bs(hy, ~Mw)
        r["x_mid"] = mean_bs(hy, Mw) - rest_mid
        e = np.flatnonzero(Mw)[-1] + 1
        Nw, okn = first_n(hy, 150, e) if e < len(hy["pa"]) else (np.zeros(len(hy["pa"]), bool), False)
        if okn:
            rest2 = mean_bs(hy, ~(Mw | Nw))
            r["x_mid_w"] = mean_bs(hy, Mw) - rest2
            r["y_within"] = mean_bs(hy, Nw) - rest2
        if nxt is not None and nxt["pa"].sum() >= 150:
            Fm, _ = first_n(nxt, 150)
            f = mean_bs(nxt, Fm)
            r["y"] = f - rest_late
            r["y_mid"] = f - rest_mid
        rows.append(r)
    return pd.DataFrame(rows)


def a11b_fit(D, AG, xcol="x", ycol="y", weights=None):
    D = D.drop(columns=[c for c in ("game_year", "age_y", "marcel", "age") if c in D])
    D = D.merge(AG, left_on=["batter", "Y"], right_on=["batter", "game_year"], how="left")
    D["age"] = D["age_y"].fillna(AG["age_y"].median())
    D = D.dropna(subset=[xcol, ycol])
    X = [D[xcol], D["age"] - 28, (D["age"] - 28) ** 2]
    names = ["b", "age", "age2"]
    if D["Y"].nunique() > 1:
        X.append((D["Y"] == 2025).astype(float)); names.append("pair")
    if weights is None:
        est, n = ols(D[ycol], X, D["batter"], names)
    else:
        est, n = wls(D[ycol], X, D[weights], D["batter"], names)
    return est["b"][0], est["b"][1], n


def a11b_all(bg, AG, nshuf=20, full=True):
    T = hs_table(bg)
    D = a11b_frame(T)
    b, se, n = a11b_fit(D, AG)
    D0s = [a11b_frame(T, hs_table(C.shuffle_games(bg, s))) for s in SEEDS_SHUF[:nshuf]]
    nulls = [a11b_fit(D0, AG)[:2] for D0 in D0s]
    bn, bnse = float(np.mean([v[0] for v in nulls])), float(np.mean([v[1] for v in nulls]))
    out = {"b": b, "se": se, "n": n, "b_null": bn, "b_null_sd": float(np.std([v[0] for v in nulls], ddof=1)),
           "excess": b - bn, "excess_se": float(np.hypot(se, bnse)), "n_shuffles": nshuf}
    if not full:
        return out
    out["mid_benchmark"] = dict(zip(["b", "se", "n"], a11b_fit(D, AG, "x_mid", "y_mid")))
    out["within_season_reference"] = dict(zip(["b", "se", "n"], a11b_fit(D.assign(yw=D["y_within"]), AG, "x_mid_w", "yw")))
    for yr in (2024, 2025):
        d = D[D["Y"] == yr]
        bb, ss, nn = a11b_fit(d, AG)
        nl = [a11b_fit(D0[D0["Y"] == yr], AG)[0] for D0 in D0s]
        out[f"pair_{yr}_{yr + 1}"] = {"b": bb, "se": ss, "n": nn, "b_null": float(np.mean(nl)), "excess": bb - float(np.mean(nl))}
    # survivorship
    D2 = D.merge(AG, left_on=["batter", "Y"], right_on=["batter", "game_year"], how="left")
    D2["age"] = D2["age_y"].fillna(AG["age_y"].median())
    MN = AG[["batter", "game_year", "marcel"]].rename(columns={"marcel": "marcel_next"})
    MN["Y"] = MN["game_year"] - 1
    D2 = D2.merge(MN[["batter", "Y", "marcel_next"]], on=["batter", "Y"], how="left")
    D2["marcel_next"] = D2["marcel_next"].fillna(D2["marcel_next"].mean())
    D2["reach"] = (D2["pa_next"] >= 150).astype(float)
    D2["xdec"] = pd.qcut(D2["x"].rank(method="first"), 10, labels=False)
    out["reach_rate_by_x_decile"] = [float(v) for v in D2.groupby("xdec")["reach"].mean()]
    out["reach_rate"] = float(D2["reach"].mean())
    Z = sm.add_constant(np.column_stack([D2["x"], D2["age"] - 28, (D2["age"] - 28) ** 2, D2["marcel_next"], D2["pa_Y"] / 100]))
    lg = sm.Logit(D2["reach"].to_numpy(), Z).fit(disp=0)
    p = lg.predict(Z)
    w = D2["reach"].mean() / p
    lo, hi = np.percentile(w[D2["reach"] == 1], [1, 99])
    D2["ipw"] = np.clip(w, lo, hi)
    out["reach_logit_x_coef"] = (float(lg.params[1]), float(lg.bse[1]))
    Dw = D2[D2["reach"] == 1]
    bw, sew, nw = a11b_fit(Dw, AG, weights="ipw")
    nlw = []
    for D0 in D0s:
        D0 = D0.merge(D2[["batter", "Y", "ipw"]], on=["batter", "Y"], how="left")
        D0 = D0[D0["ipw"].notna() & D0["y"].notna()]
        nlw.append(a11b_fit(D0, AG, weights="ipw")[:2])
    out["ipw"] = {"b": bw, "se": sew, "n": nw, "b_null": float(np.mean([v[0] for v in nlw])),
                  "excess": bw - float(np.mean([v[0] for v in nlw])),
                  "excess_se": float(np.hypot(sew, np.mean([v[1] for v in nlw])))}
    return out


# ------------------------------------------------------------------------------------------------ A11a
def a11a(bg, AG):
    T = hs_table(bg)
    rows_i, rows_ii = [], []
    for (b, y), h in T.items():
        if h["pa"].sum() < 300:
            continue
        n = len(h["pa"])
        n60 = int(np.floor(0.6 * n))
        cp = changepoint(h["bs"][:n60], h["comp"][:n60])
        if cp is not None:
            pre, post, _ = cp
            bl, _ = blocks_of(h, n60)
            for t, m, c, _ in bl:
                if t < 4:
                    rows_i.append({"batter": b, "Y": y, "h": t + 1, "dy": m - pre, "step": post - pre})
        if y < 2026:
            nxt = T.get((b, y + 1))
            cpf = changepoint(h["bs"], h["comp"])
            if cpf is not None and nxt is not None and nxt["pa"].sum() >= 100:
                pre, post, _ = cpf
                Fm, _ = first_n(nxt, 100)
                rows_ii.append({"batter": b, "Y": y, "dy": mean_bs(nxt, Fm) - pre, "step": post - pre})
    Ri, Rii = pd.DataFrame(rows_i), pd.DataFrame(rows_ii)
    Rii = Rii.merge(AG, left_on=["batter", "Y"], right_on=["batter", "game_year"], how="left")
    Rii["age"] = Rii["age_y"].fillna(AG["age_y"].median())
    cuts_i = np.quantile(Ri.drop_duplicates(["batter", "Y"])["step"].abs(), [1 / 3, 2 / 3]) if len(Ri) else [0, 0]
    cuts_ii = np.quantile(Rii["step"].abs(), [1 / 3, 2 / 3])

    def fit_i(d):
        if len(d) < 20:
            return None
        e, n = ols(d["dy"], [d["step"], (d["Y"] == 2025).astype(float), (d["Y"] == 2026).astype(float)], d["batter"], ["R", "y25", "y26"])
        return {"R": e["R"][0], "se": e["R"][1], "n": n}

    def fit_ii(d):
        if len(d) < 20:
            return None
        e, n = ols(d["dy"], [d["step"], d["age"] - 28, (d["age"] - 28) ** 2, (d["Y"] == 2025).astype(float)], d["batter"], ["R", "a", "a2", "p"])
        return {"R": e["R"][0], "se": e["R"][1], "n": n}

    out = {"within": {}, "next": {}}
    for hh in range(1, 5):
        d = Ri[Ri["h"] == hh]
        a = d["step"].abs()
        out["within"][hh] = {"pooled": fit_i(d), "tercile1": fit_i(d[a <= cuts_i[0]]),
                             "tercile2": fit_i(d[(a > cuts_i[0]) & (a <= cuts_i[1])]),
                             "tercile3": fit_i(d[a > cuts_i[1]]), "step_gt_1mph": fit_i(d[a > 1])}
    a = Rii["step"].abs()
    out["next"] = {"pooled": fit_ii(Rii), "tercile1": fit_ii(Rii[a <= cuts_ii[0]]),
                   "tercile2": fit_ii(Rii[(a > cuts_ii[0]) & (a <= cuts_ii[1])]), "tercile3": fit_ii(Rii[a > cuts_ii[1]]),
                   "step_gt_1mph": fit_ii(Rii[a > 1]), "tercile_cuts_mph": [float(c) for c in cuts_ii],
                   "share_step_gt_1mph": float((a > 1).mean())}
    out["within_tercile_cuts_mph"] = [float(c) for c in cuts_i]
    return out


# ------------------------------------------------------------------------------------------------ A11c
def a11c_pairs(bg, AG, detrend="hitter"):
    """per-hitter lag sums: within-season S[h], N[h] (h=1..10) and cross-season S2[h_in], N2[h_in] (h_in=1..20)."""
    T = hs_table(bg)
    agem = AG.set_index(["batter", "game_year"])["age_y"]
    med = AG["age_y"].median()
    by = {}
    for (b, y), h in T.items():
        if h["pa"].sum() >= 300:
            by.setdefault(b, []).append(y)
    rows = []
    for b, ys in by.items():
        if len(ys) < 2:
            continue
        for y in sorted(ys):
            h = T[(b, y)]
            bl, nc = blocks_of(h)
            a0 = agem.get((b, y), np.nan)
            a0 = med if pd.isna(a0) else a0
            jul1 = np.datetime64(f"{y}-07-01")
            for t, m, c, g0 in bl:
                age = a0 + (h["date"][g0] - jul1).astype("timedelta64[D]").astype(float) / 365.25
                rows.append((b, y, t, nc, m, c, age))
    B = pd.DataFrame(rows, columns=["batter", "Y", "t", "nY", "m", "c", "age"])
    if detrend == "hitter":
        B["x"] = np.nan
        for b, idx in B.groupby("batter").indices.items():
            d = B.iloc[idx]
            Xa = np.column_stack([np.ones(len(d)), d["age"]])
            w = np.sqrt(d["c"].to_numpy())
            beta = np.linalg.lstsq(Xa * w[:, None], d["m"].to_numpy() * w, rcond=None)[0]
            B.loc[B.index[idx], "x"] = d["m"].to_numpy() - Xa @ beta
    else:   # common quadratic age curve with hitter fixed effects
        a = B["age"] - 28
        dm = lambda s: s - s.groupby(B["batter"]).transform("mean")
        Xd = np.column_stack([dm(a), dm(a ** 2)])
        beta = np.linalg.lstsq(Xd * np.sqrt(B["c"].to_numpy())[:, None], dm(B["m"]).to_numpy() * np.sqrt(B["c"].to_numpy()), rcond=None)[0]
        B["x"] = B["m"] - (a * beta[0] + a ** 2 * beta[1])
    hitters = sorted(B["batter"].unique())
    S = np.zeros((len(hitters), 11)); N = np.zeros((len(hitters), 11))
    S2 = np.zeros((len(hitters), 21)); N2 = np.zeros((len(hitters), 21))
    for i, (b, d) in enumerate(B.groupby("batter")):
        for y, e in d.groupby("Y"):
            t, x = e["t"].to_numpy(), e["x"].to_numpy()
            dt = t[None, :] - t[:, None]
            dx2 = 0.5 * (x[None, :] - x[:, None]) ** 2
            for hh in range(1, 11):
                m = dt == hh
                S[i, hh] += dx2[m].sum(); N[i, hh] += m.sum()
            e2 = d[d["Y"] == y + 1]
            if len(e2):
                hin = (e["nY"].to_numpy()[:, None] - t[:, None]) + e2["t"].to_numpy()[None, :]
                dx2b = 0.5 * (e2["x"].to_numpy()[None, :] - x[:, None]) ** 2
                for hh in range(1, 21):
                    m = hin == hh
                    S2[i, hh] += dx2b[m].sum(); N2[i, hh] += m.sum()
    return {"S": S, "N": N, "S2": S2, "N2": N2, "n_hitters": len(hitters), "n_blocks": int(len(B))}


def vfit(g, n, model="ar_rw"):
    """WLS fit of gamma(h) = se + ss(1-phi^h) + srw h, h = 1..10. Returns dict."""
    h = np.arange(1, 11, dtype=float)
    w = np.sqrt(np.maximum(n, 1e-9))
    sc = max(g.mean(), 1e-9)
    def model_f(p):
        se, ss, phi, srw = p
        if model == "ar":
            srw = 0.0
        if model == "rw":
            ss = 0.0
        return se + ss * (1 - phi ** h) + srw * h
    res = lambda p: w * (model_f(p) - g) / sc
    best = None
    for phi0 in (0.3, 0.6, 0.85, 0.95):
        p0 = [g[0] * 0.8, max(g[-1] - g[0], 1e-3) + 0.1 * g[0], phi0, 0.01 * g[0]]
        r = least_squares(res, p0, bounds=([0, 0, 0, 0], [np.inf, np.inf, 0.995, np.inf]))
        if best is None or r.cost < best.cost:
            best = r
    se, ss, phi, srw = best.x
    if model == "ar":
        srw = 0.0
    if model == "rw":
        ss, phi = 0.0, np.nan
    hl = 50 * np.log(0.5) / np.log(phi) if (phi is not None and np.isfinite(phi) and 0 < phi < 1) else np.nan
    return {"sigma2_e": float(se), "sigma2_s": float(ss), "phi": float(phi), "sigma2_rw": float(srw), "half_life_pa": float(hl),
            "wsse": float(2 * best.cost * sc * sc), "fitted": [float(v) for v in model_f(best.x)]}


def cfit(g2, n2, P):
    """cross-season: gamma = se + ss(1 - phi^h rho) + srw h + soff, with within-season parameters fixed."""
    h = np.arange(1, 21, dtype=float)
    ok = n2 > 0
    if ok.sum() < 3:
        return {"rho_off": np.nan, "sigma2_off": np.nan}
    se, ss, phi, srw = P["sigma2_e"], P["sigma2_s"], P["phi"], P["sigma2_rw"]
    phi = 0.0 if not np.isfinite(phi) else phi
    w = np.sqrt(n2[ok]); sc = max(g2[ok].mean(), 1e-9)
    f = lambda p: se + ss * (1 - phi ** h[ok] * p[0]) + srw * h[ok] + p[1]
    r = least_squares(lambda p: w * (f(p) - g2[ok]) / sc, [0.5, 0.05 * g2[ok].mean()], bounds=([0, 0], [1, np.inf]))
    return {"rho_off": float(r.x[0]), "sigma2_off": float(r.x[1])}


def a11c(bg, AG, boot=0, detrend="hitter"):
    P = a11c_pairs(bg, AG, detrend)
    def fit_all(wt):
        S, N = (P["S"] * wt[:, None]).sum(0), (P["N"] * wt[:, None]).sum(0)
        S2, N2 = (P["S2"] * wt[:, None]).sum(0), (P["N2"] * wt[:, None]).sum(0)
        g, n = S[1:] / N[1:], N[1:]
        g2, n2 = np.where(N2[1:] > 0, S2[1:] / np.maximum(N2[1:], 1), np.nan), N2[1:]
        main = vfit(g, n)
        return main, g, n, g2, n2
    wt = np.ones(P["n_hitters"])
    main, g, n, g2, n2 = fit_all(wt)
    out = {"ar_rw": main, "ar_only": vfit(g, n, "ar"), "rw_only": vfit(g, n, "rw"), "gamma": [float(v) for v in g],
           "pairs": [int(v) for v in n], "cross": cfit(g2, n2, main), "gamma_cross": [None if not np.isfinite(v) else float(v) for v in g2],
           "pairs_cross": [int(v) for v in n2], "n_hitters": P["n_hitters"], "n_blocks": P["n_blocks"]}
    if boot:
        rng = np.random.default_rng(SEED_BOOT)
        B = []
        for _ in range(boot):
            wt = rng.multinomial(P["n_hitters"], np.ones(P["n_hitters"]) / P["n_hitters"]).astype(float)
            m, g_, n_, g2_, n2_ = fit_all(wt)
            cr = cfit(g2_, n2_, m)
            B.append([m["phi"], m["half_life_pa"], m["sigma2_rw"], m["sigma2_s"], m["sigma2_e"], cr["rho_off"], cr["sigma2_off"]])
        B = np.array(B)
        names = ["phi", "half_life_pa", "sigma2_rw", "sigma2_s", "sigma2_e", "rho_off", "sigma2_off"]
        out["boot"] = {k: {"lo": float(np.nanpercentile(B[:, i], 2.5)), "hi": float(np.nanpercentile(B[:, i], 97.5)),
                           "share_nan": float(np.isnan(B[:, i]).mean())} for i, k in enumerate(names)}
        out["boot"]["phi_at_bound_share"] = float((B[:, 0] >= 0.9949).mean())
        out["boot"]["n_draws"] = boot
    return out


# ------------------------------------------------------------------------------------------------ A1 bits on any panel
def a1_decay(bg, lg, nshuf=3, seed0=101):
    bg0s = [C.shuffle_games(bg, seed0 + i) for i in range(nshuf)]
    keep = set(C.windows(bg, 50, 200, lg)["row"])
    keep0 = [set(C.windows(b0, 50, 200, lg)["row"]) for b0 in bg0s]
    out = {}
    for fwd in [25, 50, 100, 150, 200]:
        W = C.windows(bg, 50, fwd, lg); W = W[W["row"].isin(keep)]
        W0s = []
        for b0, k0 in zip(bg0s, keep0):
            x = C.windows(b0, 50, fwd, lg); W0s.append(x[x["row"].isin(k0)])
        p = C.persistence(W, W0s, ["bat_speed"])["bat_speed"]
        out[fwd] = {"persistence": p["persistence"], "slope_null": p["slope_null"]}
    return out


def p5050(bg, lg, nshuf=3, seed0=101):
    W = C.windows(bg, 50, 50, lg)
    W0s = [C.windows(C.shuffle_games(bg, seed0 + i), 50, 50, lg) for i in range(nshuf)]
    return C.persistence(W, W0s, ["bat_speed"])["bat_speed"]["persistence"]


def jump_share(bg):
    T = hs_table(bg)
    j = []
    for h in T.values():
        cp = changepoint(h["bs"], h["comp"])
        if cp is not None:
            j.append(abs(cp[1] - cp[0]))
    return np.array(j)


# ------------------------------------------------------------------------------------------------ noise moments
def noise_moments(bg):
    def build():
        d = C.pitches(DATA)
        d = d[d["comp"] == 1]
        bs = pd.to_numeric(d["bat_speed"], errors="coerce")
        g = bs.groupby([d["batter"], d["game_pk"]])
        ss = ((bs - g.transform("mean")) ** 2).groupby([d["batter"], d["game_pk"]]).sum()
        dfree = (g.count() - 1).clip(lower=0)
        return float(np.sqrt(ss.sum() / dfree.sum()))
    sigma_sw = A3.cached("a11_sigma_sw", build)
    T = hs_table(bg)
    dev, samp, cov = [], [], []
    for h in T.values():
        if h["pa"].sum() < 300:
            continue
        q = h["comp"] >= 5
        y = h["bs"][q] / h["comp"][q]
        mu = h["bs"][q].sum() / h["comp"][q].sum()
        e = y - mu
        dev.append(e); samp.append(sigma_sw ** 2 / h["comp"][q]); cov.append(e[1:] * e[:-1])
    dev, samp, cov = np.concatenate(dev), np.concatenate(samp), np.concatenate(cov)
    sg2 = max(float(np.mean(dev ** 2) - np.mean(samp) - np.mean(cov)), 0.0)
    return {"sigma_sw": sigma_sw, "sigma_g": float(np.sqrt(sg2)), "var_game_dev": float(np.mean(dev ** 2)),
            "mean_sampling_var": float(np.mean(samp)), "lag1_autocov": float(np.mean(cov))}


def season_change_var(bg):
    T = hs_table(bg)
    d = []
    for (b, y), h in T.items():
        n = T.get((b, y + 1))
        if n is not None and h["pa"].sum() >= 300 and n["pa"].sum() >= 300:
            d.append(n["bs"].sum() / n["comp"].sum() - h["bs"].sum() / h["comp"].sum())
    return float(np.var(d)), len(d)


# ------------------------------------------------------------------------------------------------ runs
def run_real():
    bg, lg = A3.panel()
    AG = ages()
    chk = C.slope(C.windows(bg, 50, 50, lg), "bat_speed")[0]
    log("rebuild check slope", chk)
    out = {"rebuild_check": {"bat_speed_actual_slope_50_50": chk, "published": 0.37419380306389444,
                             "match_1e-6": abs(chk - 0.37419380306389444) < 1e-6}}
    j = jump_share(bg)
    out["a1_changepoint_check"] = {"n": int(len(j)), "share_over_1mph": float((j > 1).mean()), "median": float(np.median(j))}
    log("A11b"); out["A11b"] = a11b_all(bg, AG, 20, full=True); log(json.dumps({k: v for k, v in out["A11b"].items() if not isinstance(v, (dict, list))}))
    log("A11a"); out["A11a"] = a11a(bg, AG); log(out["A11a"]["next"]["pooled"], [out["A11a"]["within"][h]["pooled"] for h in range(1, 5)])
    log("A11c"); out["A11c"] = a11c(bg, AG, boot=500)
    out["A11c_common_age_curve"] = a11c(bg, AG, boot=0, detrend="common")
    log(out["A11c"]["ar_rw"], out["A11c"]["cross"], out["A11c"]["boot"])
    out["noise"] = noise_moments(bg)
    out["season_change_var"] = season_change_var(bg)
    log("noise", out["noise"])
    dump("a11_real", out)
    return out


def calibrate(f, lo, hi, target=TARGET_P, tol=0.004, it=12):
    """monotone bisection on log scale; f(scale) -> persistence."""
    flo, fhi = f(lo), f(hi)
    tried = [(lo, flo), (hi, fhi)]
    for _ in range(it):
        if not (flo <= target <= fhi):
            break
        mid = np.sqrt(lo * hi)
        fm = f(mid); tried.append((mid, fm))
        if abs(fm - target) < tol:
            return mid, fm, tried
        if fm < target:
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    best = min(tried, key=lambda t: abs(t[1] - target))
    return best[0], best[1], tried


def run_sims(nrep):
    bg, lg = A3.panel()
    AG = ages()
    real = json.load(open(os.path.join(OUT, "a11_real.json")))
    nz = real["noise"]
    bgs, skel = SIM.a11_skeleton(bg)
    mu_map = bgs.groupby("batter")["bs_num"].sum() / bgs.groupby("batter")["comp"].sum()
    mu = bgs["batter"].map(mu_map).to_numpy()
    jumps = jump_share(bg)
    var_real, _ = season_change_var(bg)
    worlds = {"a_none": {}, "b_hl50": {"hl": 50}, "b_hl150": {"hl": 150}, "b_hl400": {"hl": 400}, "c_step": {"step": True},
              "d_mix": {"hl": 150, "step": True}}
    # ---- calibration on replicate 0 (common random numbers)
    D0 = SIM.a11_draws(skel, len(bgs), jumps, SEED_SIM)
    ars0 = {hl: SIM.a11_ar(skel, D0, hl) for hl in (50, 150, 400)}
    base = dict(mu=mu, sigma_sw=nz["sigma_sw"], sigma_g=nz["sigma_g"])
    cal = {}
    for wname, spec in worlds.items():
        if wname == "a_none":
            cal[wname] = {"ar_sd": 0.0, "step_k": 0.0}
        elif wname.startswith("b_"):
            f = lambda s, hl=spec["hl"]: p5050(SIM.a11_world(bgs, D0, ar=ars0[hl], ar_sd=s, **base), lg)
            s, p, tr = calibrate(f, 0.2, 4.0)
            cal[wname] = {"ar_sd": s, "step_k": 0.0, "p5050": p, "tried": tr}
        elif wname == "c_step":
            f = lambda k: p5050(SIM.a11_world(bgs, D0, step_k=k, **base), lg)
            k, p, tr = calibrate(f, 0.1, 4.0)
            cal[wname] = {"ar_sd": 0.0, "step_k": k, "p5050": p, "tried": tr}
        else:
            sb, kc = cal["b_hl150"]["ar_sd"], cal["c_step"]["step_k"]
            f = lambda a: p5050(SIM.a11_world(bgs, D0, ar=ars0[150], ar_sd=a * sb * np.sqrt(.5), step_k=a * kc * np.sqrt(.5), **base), lg)
            a, p, tr = calibrate(f, 0.5, 2.0)
            cal[wname] = {"ar_sd": a * sb * np.sqrt(.5), "step_k": a * kc * np.sqrt(.5), "common": a, "p5050": p, "tried": tr}
        # between-season change variance
        w0 = SIM.a11_world(bgs, D0, ar=ars0.get(spec.get("hl")), ar_sd=cal[wname]["ar_sd"], step_k=cal[wname]["step_k"], **base)
        v0, _ = season_change_var(w0)
        cal[wname]["sigma_delta"] = float(np.sqrt(max(var_real - v0, 0.0)))
        log("calibrated", wname, {k: v for k, v in cal[wname].items() if k != "tried"})
    dump("a11_sims_calibration", {"cal": cal, "noise": nz, "var_season_change_real": var_real})
    # ---- replicates
    res = {w: [] for w in worlds}
    for r in range(nrep):
        D = D0 if r == 0 else SIM.a11_draws(skel, len(bgs), jumps, SEED_SIM + r)
        ars = ars0 if r == 0 else {hl: SIM.a11_ar(skel, D, hl) for hl in (50, 150, 400)}
        for wname, spec in worlds.items():
            c = cal[wname]
            w = SIM.a11_world(bgs, D, ar=ars.get(spec.get("hl")), ar_sd=c["ar_sd"], step_k=c["step_k"],
                              sigma_delta=c["sigma_delta"], **base)
            t = time.time()
            o = {"p5050": p5050(w, lg), "decay": a1_decay(w, lg)}
            j = jump_share(w)
            o["share_jump_over_1mph"] = float((j > 1).mean())
            o["A11b"] = a11b_all(w, AG, 20, full=False)
            o["A11a"] = a11a(w, AG)
            o["A11c"] = a11c(w, AG, boot=0)
            res[wname].append(o)
            log("rep", r, wname, f"{time.time() - t:.0f}s", round(o["p5050"], 3), "b-null", round(o["A11b"]["excess"], 3),
                "Rnext", round(o["A11a"]["next"]["pooled"]["R"], 3), "HL", round(o["A11c"]["ar_rw"]["half_life_pa"], 1),
                "srw", round(o["A11c"]["ar_rw"]["sigma2_rw"], 4))
        dump("a11_sims", {"cal": cal, "replicates": res})
    return {"cal": cal, "replicates": res}


# ------------------------------------------------------------------------------------------------ merge + decision rule
def summarise_world(reps):
    def g(f):
        v = np.array([f(o) for o in reps], float)
        return {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)) if len(v) > 1 else None}
    s = {"p5050": g(lambda o: o["p5050"]), "share_jump_over_1mph": g(lambda o: o["share_jump_over_1mph"]),
         "b": g(lambda o: o["A11b"]["b"]), "b_null": g(lambda o: o["A11b"]["b_null"]), "b_excess": g(lambda o: o["A11b"]["excess"]),
         "R_next": g(lambda o: o["A11a"]["next"]["pooled"]["R"]),
         "half_life_pa": g(lambda o: o["A11c"]["ar_rw"]["half_life_pa"]), "phi": g(lambda o: o["A11c"]["ar_rw"]["phi"]),
         "sigma2_rw": g(lambda o: o["A11c"]["ar_rw"]["sigma2_rw"]), "rho_off": g(lambda o: o["A11c"]["cross"]["rho_off"]),
         "sigma2_off": g(lambda o: o["A11c"]["cross"]["sigma2_off"])}
    for fwd in ("25", "50", "100", "150", "200"):
        s[f"decay_{fwd}"] = g(lambda o, f=fwd: o["decay"][f]["persistence"])
    for h in ("1", "2", "3", "4"):
        s[f"R_within_{h}"] = g(lambda o, h=h: o["A11a"]["within"][h]["pooled"]["R"])
    return s


def classify(real_b, real_R, sig, hl_finite, srw_lo_pos):
    """real_b / real_R: (est, se). sig: world summaries. Returns branch and the per-world checks."""
    ci = lambda e: (e[0] - 1.96 * e[1], e[0] + 1.96 * e[1])
    cb, cR = ci(real_b), ci(real_R)
    cons = lambda c, W, key: c[0] <= sig[W][key]["mean"] <= c[1]
    chk = {W: {"b_consistent": cons(cb, W, "b"), "R_next_consistent": cons(cR, W, "R_next")} for W in sig}
    permanent = (chk["c_step"]["b_consistent"] and chk["c_step"]["R_next_consistent"]
                 and not any(chk[W]["b_consistent"] or chk[W]["R_next_consistent"] for W in ("b_hl50", "b_hl150")) and srw_lo_pos)
    slow = (not chk["c_step"]["b_consistent"]) and (not chk["c_step"]["R_next_consistent"]) and hl_finite
    between = lambda e, key: min(sig["b_hl150"][key]["mean"], sig["c_step"][key]["mean"]) <= e[0] <= max(sig["b_hl150"][key]["mean"], sig["c_step"][key]["mean"])
    dist = {W: abs(real_b[0] - sig[W]["b"]["mean"]) / real_b[1] + abs(real_R[0] - sig[W]["R_next"]["mean"]) / real_R[1] for W in sig}
    mixed = between(real_b, "b") and between(real_R, "R_next") and min(dist, key=dist.get) == "d_mix"
    branch = "permanent" if permanent else ("slow_reverting" if slow else ("mixed" if mixed else "undetermined"))
    return {"branch": branch, "ci_b": cb, "ci_R_next": cR, "checks": chk, "distance_to_world": dist,
            "closest_world": min(dist, key=dist.get)}


def merge():
    real = json.load(open(os.path.join(OUT, "a11_real.json")))
    sims = json.load(open(os.path.join(OUT, "a11_sims.json")))
    sig = {W: summarise_world(r) for W, r in sims["replicates"].items()}
    b, R, c = real["A11b"], real["A11a"]["next"]["pooled"], real["A11c"]
    hl_finite = c["boot"]["phi"]["hi"] < 0.9949
    srw_lo_pos = c["boot"]["sigma2_rw"]["lo"] > 1e-6
    prim = classify((b["b"], b["se"]), (R["R"], R["se"]), sig, hl_finite, srw_lo_pos)
    ipw = classify((b["ipw"]["b"], b["ipw"]["se"]), (R["R"], R["se"]), sig, hl_finite, srw_lo_pos)
    # sensitivity (not the prespecified rule): b - b_null in place of b
    sig_x = {W: dict(s, b=s["b_excess"]) for W, s in sig.items()}
    exc = classify((b["excess"], b["excess_se"]), (R["R"], R["se"]), sig_x, hl_finite, srw_lo_pos)
    same_sign = np.sign(b["pair_2024_2025"]["excess"]) == np.sign(b["pair_2025_2026"]["excess"])
    robust = {"ipw_same_branch": ipw["branch"] == prim["branch"], "pairs_same_sign": bool(same_sign)}
    final = prim["branch"] if (robust["ipw_same_branch"] and robust["pairs_same_sign"]) else "undetermined"
    real_sig = {"p5050": 0.2598, "share_jump_over_1mph": real["a1_changepoint_check"]["share_over_1mph"],
                "b": b["b"], "b_null": b["b_null"], "b_excess": b["excess"], "R_next": R["R"],
                "half_life_pa": c["ar_rw"]["half_life_pa"], "phi": c["ar_rw"]["phi"], "sigma2_rw": c["ar_rw"]["sigma2_rw"],
                "rho_off": c["cross"]["rho_off"], "sigma2_off": c["cross"]["sigma2_off"]}
    a1 = json.load(open(os.path.join(HERE, "results", "a1.json")))
    for fwd in ("25", "50", "100", "150", "200"):
        real_sig[f"decay_{fwd}"] = a1["horizons_fixed_sample"][fwd]["bat_speed"]["persistence"]
    for h in ("1", "2", "3", "4"):
        real_sig[f"R_within_{h}"] = real["A11a"]["within"][h]["pooled"]["R"]
    yearly = [os.path.join(DATA, f"statcast_{y}.parquet") for y in (2024, 2025, 2026)]
    out = {"meta": {"plan": "PLAN_ADDENDUM_2026-10-01c.md",
                    # the plan file is kept with the authors; its recorded hash is used when it is not alongside the repo
                    "plan_sha16": sha16(PLAN) if os.path.exists(PLAN) else "210effb2a9bd81ce", "built": time.strftime("%Y-%m-%d %H:%M"),
                    "statcast_sha16": ({y: sha16(p) for y, p in zip((2024, 2025, 2026), yearly)} if all(map(os.path.exists, yearly)) else
                                       {f: sha16(os.path.join(DATA, f)) for f in sorted(os.listdir(DATA)) if f.endswith(".parquet")}),
                    "statcast_source": "Statcast_Shared statcast_YYYY.parquet (regular-season rows, fetch_statcast KEEP columns); "
                                       "shared-cache files 65b82081ea38f1ea / 3c686658c711097d / 88660e24fe4a7b00",
                    "seeds": {"shuffles": SEEDS_SHUF, "bootstrap": SEED_BOOT, "simulations": SEED_SIM},
                    "n_sim_replicates": len(next(iter(sims["replicates"].values())))},
           "real": real, "signature": {"real": real_sig, **sig}, "calibration": sims["cal"],
           "decision": {"primary": prim, "ipw": ipw, "robustness": robust, "branch": final,
                        "sensitivity_excess_not_prespecified": exc,
                        "half_life_ci_finite": bool(hl_finite), "sigma2_rw_lower_bound_positive": bool(srw_lo_pos)},
           "sim_replicates": sims["replicates"]}
    dump("a11", out)
    print(json.dumps(out["decision"], indent=1, default=str))
    return out


if __name__ == "__main__":
    t0 = time.time()
    if "real" in PARTS:
        run_real()
    if "sims" in PARTS:
        run_sims(NREP)
    if "merge" in PARTS:
        merge()
    log(f"done in {time.time() - t0:.0f}s")
