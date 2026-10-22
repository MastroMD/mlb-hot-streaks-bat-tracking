"""AJSM game-level analyses F4a-F4e (plan addendum F, PLAN_ADDENDUM_2026-10-05f.md, written before this ran).

python3 ajsm_game.py DATA_DIR IL_FILE OUT_JSON [--shuffles K] [--coded CODED_CSV] [--il-original ORIGINAL_IL_FILE]

DATA_DIR  Statcast regular-season files (the 22 monthly fetch_statcast.py files or the three season files from
          paper/tools/rebuild_data_dir.py). Only the columns below are read, so the run fits in about 3 GB of memory.
IL_FILE   repo/data_public/il_transactions_hitters_2024_2026_corrected.txt (addendum G; the original file is
          il_transactions_hitters_2024_2026.txt, given with --il-original for the A8 comparison)
CODED_CSV repo/data_public/il_placements_coded_2023_2026.csv (region, mechanism, impact; addendum G4)

F4a  bat speed in the 30 days before an IL placement against control dates, and A8 re-run on at-risk decision points
F4b  bat speed (and contact quality) in the first 40 games after return against matched controls
F4c  split-half reliability, SEM and minimal detectable change of 25-, 50- and 100-PA windows
F4d  bat speed adjusted for count and zone: persistence and reliability
F4e  faster and slower recent bat speed as separate terms for oblique and any placement (post hoc)

Shares the park-season and week-of-season adjustment, the competitive-swing filter and the window builder with
analysis3.py (common3.py, hot_streak_study.py), so every definition matches the paper.
"""
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common3 as C            # noqa: E402
import hot_streak_study as H   # noqa: E402

opt = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
_skip = {sys.argv.index(k) + 1 for k in ("--shuffles", "--coded", "--il-original") if k in sys.argv}
args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and i not in _skip]
DATA, ILF, OUT = args[:3]
CODED, IL_ORIG = opt("--coded"), opt("--il-original")
K_SHUF = int(sys.argv[sys.argv.index("--shuffles") + 1]) if "--shuffles" in sys.argv else 5
SEED = 505
FAM = {"O": "oblique", "H": "hamstring", "W": "hand_wrist", "X": "other"}
FAMS = ["oblique", "hamstring", "hand_wrist", "other"]
COLS = ["game_date", "game_pk", "game_type", "game_year", "at_bat_number", "pitch_number", "batter", "pitcher", "stand",
        "p_throws", "home_team", "away_team", "balls", "strikes", "zone", "description", "events", "type",
        "release_speed", "launch_speed", "launch_angle", "launch_speed_angle", "estimated_woba_using_speedangle",
        "woba_value", "woba_denom", "bat_speed", "swing_length"]
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
sha16 = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


# ------------------------------------------------------------------ data
def load(data_dir):
    import re
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    # a folder of full-width season files (Statcast_Shared) may hold other seasons: keep 2024-2026 by file name
    files = [f for f in files if not re.findall(r"20\d\d", os.path.basename(f))
             or any(y in ("2024", "2025", "2026") for y in re.findall(r"20\d\d", os.path.basename(f)))]
    if not files:
        sys.exit(f"no parquet files in {data_dir}")
    dfs = []
    for f in files:
        import pyarrow.parquet as pq
        have = set(pq.ParquetFile(f).schema_arrow.names)
        dfs.append(pd.read_parquet(f, columns=[c for c in COLS if c in have]))
    df = pd.concat(dfs, ignore_index=True)
    for c in COLS:
        if c not in df:
            df[c] = np.nan
    df = df[(df["game_type"] == "R") & pd.to_numeric(df["game_year"], errors="coerce").between(2024, 2026)].copy()
    for c in df.columns:
        dt = str(df[c].dtype)
        if dt in ("Int8", "Int16", "Int32", "Int64", "Float32", "Float64", "UInt8", "UInt16", "UInt32", "UInt64"):
            df[c] = df[c].astype("float64")
        elif dt in ("string", "str") or dt.startswith("string"):
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    for c in ("batter", "pitcher", "game_pk", "game_year"):
        df[c] = df[c].astype("int64")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    return df, files


def il_events(path):
    P, A = [], []
    for line in open(path):
        b, ev = line.strip().split(":")
        for e in ev.split():
            d = pd.Timestamp("20" + e[:6])
            if e[6] == "P":
                P.append({"batter": int(b), "start": d, "fam": FAM[e[7]]})
            else:
                A.append({"batter": int(b), "end": d})
    P = pd.DataFrame(P).sort_values(["batter", "start"]).reset_index(drop=True)
    A = pd.DataFrame(A).sort_values(["batter", "end"]).reset_index(drop=True)
    acts = []
    for b, g in P.groupby("batter", sort=False):
        a = A.loc[A["batter"] == b, "end"].tolist()
        st = g["start"].tolist()
        for k, s in enumerate(st):
            lim = st[k + 1] if k + 1 < len(st) else pd.Timestamp("2100-01-01")
            c = [x for x in a if s <= x <= lim]
            acts.append(c[0] if c else pd.NaT)
    P["ret"] = acts
    return P, A


raw, FILES = load(DATA)
log("pitches", len(raw))
d = C.adjust(raw)
d = H.pitch_features(d)
d = H.context(d)
del raw
# count-and-zone adjusted bat speed on the same competitive swings (F4d)
bsr = np.where(d["comp"] == 1, d["bs_num"], np.nan)
grp = [pd.to_numeric(d["balls"], errors="coerce").fillna(-1).clip(upper=3), pd.to_numeric(d["strikes"], errors="coerce").fillna(-1).clip(upper=2),
       d["inzone"]]
cell = pd.Series(bsr).groupby(grp).transform("mean")
d["bsc_num"] = np.where(d["comp"] == 1, bsr - cell + np.nanmean(bsr), 0.0)
# every tracked swing, no competitive filter (sensitivity: the season-p10 filter truncates and so attenuates real declines)
_bs, _sl = pd.to_numeric(d["bat_speed"], errors="coerce"), pd.to_numeric(d["swing_length"], errors="coerce")
_tr = _bs.notna() & _sl.notna()
d["compall"] = _tr.astype(np.int8)
d["bsall_num"] = np.where(_tr, _bs, 0.0)
d["n_pitch"] = 1
KEYS = ["batter", "game_year", "game_date", "game_pk"]
SUMS = C.SUM_COLS + ["bsc_num", "compall", "bsall_num", "n_pitch"]
bg = d.groupby(KEYS, sort=False)[SUMS].sum().reset_index().sort_values(["batter", "game_date", "game_pk"]).reset_index(drop=True)
bg = bg[bg["pa"] > 0].reset_index(drop=True)
lg = H.league_means(bg)
# swing-, ball-in-play- and PA-level units for reliability (F4c)
SW = d.loc[d["comp"] == 1, KEYS + ["at_bat_number", "pitch_number", "bs_num", "sl_num", "fast", "bsc_num"]].copy()
BIP = d.loc[d["bip"] == 1, KEYS + ["at_bat_number", "ev_num", "hard"]].copy()
PA = d.loc[d["pa"] == 1, KEYS + ["at_bat_number", "woba_num", "xwoba_num"]].copy()
n_pitches, n_hitters = int(len(d)), int(d["batter"].nunique())
del d
P, A = il_events(ILF)
P["imp"] = "unknown"
if CODED:
    _cd = pd.read_csv(CODED, parse_dates=["day"]).drop_duplicates(["batter", "day"])
    P = P.merge(_cd[["batter", "day", "impact"]], left_on=["batter", "start"], right_on=["batter", "day"], how="left").drop(columns="day")
    P["imp"] = np.where(P["impact"].isna(), "unknown", np.where(P["impact"] == True, "impact", "non_impact"))  # noqa: E712
LABEL_END = pd.Timestamp(bg["game_date"].max())
# IL universe (addendum F): the file lists only hitters with a >=300-PA (wOBA denominator) season in 2024-2026, so a
# hitter outside that set has no recorded placements whether or not he was injured. Control periods, control hitters and
# the at-risk models are restricted to hitters inside it (fixed 2026-10-06 after the independent check; earlier runs
# used every hitter, which labels uncovered hitters' injuries as non-events).
COVERED = set(bg.groupby(["batter", "game_year"])["pa"].sum().loc[lambda x: x >= 300].index.get_level_values(0))
log("batter-games", len(bg), "placements", len(P), "label end", LABEL_END.date(), "covered hitters", len(COVERED))
out = {"plan": {"file": "PLAN_ADDENDUM_2026-10-05f.md", "sha16": "b21ecb192b5903d0", "addendum_g": "PLAN_ADDENDUM_2026-10-05g.md", "addendum_g_sha16": "95404f233ccc6d1a"},
       "inputs": {"statcast_files": len(FILES), "il_sha16": sha16(ILF), "n_pitches": n_pitches, "n_hitters": n_hitters,
                  "batter_games": int(len(bg)), "placements": int(len(P)), "label_end": str(LABEL_END.date())},
       "n_shuffles": K_SHUF}


def wmean(num, den):
    return num / den if den > 0 else np.nan


def ols_mean(y, w, g):
    """Weighted mean with hitter-clustered SE (intercept-only WLS)."""
    y, w, g = np.asarray(y, float), np.asarray(w, float), np.asarray(g)
    ok = np.isfinite(y) & (w > 0)
    f = sm.WLS(y[ok], np.ones(ok.sum()), weights=w[ok]).fit(cov_type="cluster", cov_kwds={"groups": g[ok]})
    return {"mean": float(f.params[0]), "se": float(f.bse[0]), "lo": float(f.conf_int()[0][0]),
            "hi": float(f.conf_int()[0][1]), "p": float(f.pvalues[0]), "n": int(ok.sum()), "hitters": int(len(set(g[ok])))}


# per hitter-season arrays for fast window sums
G = {}
for (b, y), idx in bg.groupby(["batter", "game_year"]).indices.items():
    x = bg.iloc[idx]
    G[(b, y)] = {"t": x["game_date"].to_numpy().astype("datetime64[D]"),
                 **{c: x[c].to_numpy(float) for c in ("comp", "bs_num", "sl_num", "fast", "bip", "ev_num", "hard", "pa", "xwoba_num", "woba_num", "bsc_num",
                                                "compall", "bsall_num")}}
MET = {"bat_speed": ("bs_num", "comp", 5), "bat_speed_all_swings": ("bsall_num", "compall", 5), "swing_len": ("sl_num", "comp", 5), "fast_swing": ("fast", "comp", 5),
       "ev": ("ev_num", "bip", 3), "hard_hit": ("hard", "bip", 3), "xwoba": ("xwoba_num", "pa", 5)}


def baseline(b, y, before):
    """Same-season games before `before` (>= 90 competitive swings), else the previous season. Returns sums and source."""
    g = G.get((b, y))
    if g is not None:
        m = g["t"] < np.datetime64(before, "D")
        if g["comp"][m].sum() >= 90:
            return {c: g[c][m].sum() for c in g if c != "t"}, "same"
    g = G.get((b, y - 1))
    if g is not None and g["comp"].sum() >= 90:
        return {c: g[c].sum() for c in g if c != "t"}, "prev"
    return None, None


def dev(sums, base, metric):
    n, dn, _ = MET[metric]
    return wmean(sums[n], sums[dn]) - wmean(base[n], base[dn]), sums[dn]


# ------------------------------------------------------------------ F4a pre-placement trajectory
WIN = [(1, 3), (4, 7), (8, 14), (15, 21), (22, 30), (1, 14)]


def pre_rows(b, y, S, inj, fam, imp="control"):
    base, src = baseline(b, y, S - pd.Timedelta(days=30))
    if base is None:
        return []
    g = G[(b, y)]
    rows = []
    for lo, hi in WIN:
        m = (g["t"] >= np.datetime64(S - pd.Timedelta(days=hi), "D")) & (g["t"] <= np.datetime64(S - pd.Timedelta(days=lo), "D"))
        s = {c: g[c][m].sum() for c in g if c != "t"}
        need = 20 if (lo, hi) == (1, 14) else 5
        if s["comp"] < need:
            continue
        v, w = dev(s, base, "bat_speed")
        va, wa = dev(s, base, "bat_speed_all_swings") if s["compall"] >= need else (np.nan, 0.0)
        rows.append({"batter": b, "win": f"{lo}-{hi}", "dev": v, "w": w, "dev_all": va, "w_all": wa, "inj": inj, "fam": fam,
                     "imp": imp, "src": src, "month": S.month, "year": y, "date": S})
    return rows


rows = []
for p in P.itertuples():
    if (p.batter, p.start.year) in G and p.start <= LABEL_END:
        rows += pre_rows(p.batter, p.start.year, p.start, 1, p.fam, p.imp)
Pb = {b: g["start"].to_numpy() for b, g in P.groupby("batter")}
Ab = {b: g["end"].to_numpy() for b, g in A.groupby("batter")}
rng = np.random.default_rng(SEED)
for (b, y), g in G.items():
    if b not in COVERED:
        continue
    ts = g["t"]
    pl = Pb.get(b, np.array([], dtype="datetime64[ns]")).astype("datetime64[D]")
    ac = Ab.get(b, np.array([], dtype="datetime64[ns]")).astype("datetime64[D]")
    for t in ts[::3]:                                    # every third game date keeps the control set manageable
        if ((pl >= t) & (pl <= t + np.timedelta64(60, "D"))).any() or ((ac <= t) & (ac >= t - np.timedelta64(45, "D"))).any():
            continue
        if pd.Timestamp(t) + pd.Timedelta(days=60) > LABEL_END:
            continue
        rows += pre_rows(b, y, pd.Timestamp(t), 0, "control")
R = pd.DataFrame(rows)
F4a = {"n_placements_with_rows": int(R.loc[R["inj"] == 1, ["batter", "date"]].drop_duplicates().shape[0]),
       "n_control_dates": int(R.loc[R["inj"] == 0, ["batter", "date"]].drop_duplicates().shape[0]),
       "n_control_hitters": int(R.loc[R["inj"] == 0, "batter"].nunique()), "windows": {}}
GROUPS = [("all", None, None)] + [(f, "fam", f) for f in FAMS] + [("impact", "imp", "impact"), ("non_impact", "imp", "non_impact")]
for fam, gcol, gval in GROUPS:
    F4a["windows"][fam] = {}
    for w_ in [f"{lo}-{hi}" for lo, hi in WIN]:
        D = R[(R["win"] == w_) & ((R["inj"] == 0) | ((R[gcol] == gval) if gcol else True))]
        if D["inj"].sum() < 5:
            continue
        X = pd.get_dummies(D[["month", "src"]].astype(str), drop_first=True, dtype=float)
        X.insert(0, "inj", D["inj"].astype(float))
        f = sm.WLS(D["dev"], sm.add_constant(X), weights=D["w"]).fit(cov_type="cluster", cov_kwds={"groups": D["batter"]})
        ci90 = f.conf_int(alpha=0.10).loc["inj"]
        Da = D[D["dev_all"].notna() & (D["w_all"] > 0)]
        fa = sm.WLS(Da["dev_all"], sm.add_constant(X.loc[Da.index]), weights=Da["w_all"]).fit(cov_type="cluster", cov_kwds={"groups": Da["batter"]})
        F4a["windows"][fam][w_] = {"diff_all_swings": float(fa.params["inj"]), "se_all_swings": float(fa.bse["inj"]),
                                    "diff": float(f.params["inj"]), "se": float(f.bse["inj"]), "p": float(f.pvalues["inj"]),
                                    "lo": float(f.conf_int().loc["inj", 0]), "hi": float(f.conf_int().loc["inj", 1]),
                                    "ci90": [float(ci90[0]), float(ci90[1])], "n_inj": int(D["inj"].sum()),
                                    "equivalent_0p5": bool(ci90[0] > -0.5 and ci90[1] < 0.5),
                                    "inj_raw_mean": float(np.average(D.loc[D["inj"] == 1, "dev"], weights=D.loc[D["inj"] == 1, "w"]))}
out["F4a"] = F4a
log("F4a done", json.dumps(F4a["windows"]["all"].get("1-14", {})))


# ------------------------------------------------------------------ F4a at-risk A8 and F4e
def a8(at_risk, min_season_pa, label, P=P):
    W = C.windows(bg, 50, 50, lg, at_risk=at_risk)
    W = W[(W["season_pa"] >= min_season_pa) & W["batter"].isin(COVERED)].copy()
    W["game_date"] = pd.to_datetime(W["game_date"])
    W = W[W["game_date"] + pd.Timedelta(days=30) <= LABEL_END]
    il_next, il_fam, ret = np.zeros(len(W), int), np.array([""] * len(W), dtype=object), np.zeros(len(W), int)
    Pg = {b: g for b, g in P.groupby("batter")}
    for n, (dt, b) in enumerate(zip(W["game_date"], W["batter"])):
        g = Pg.get(b)
        if g is not None:
            h = g[(g["start"] >= dt) & (g["start"] <= dt + pd.Timedelta(days=30))]
            if len(h):
                il_next[n] = 1; il_fam[n] = h["fam"].iloc[0]
        a = Ab.get(b)
        if a is not None and ((a <= np.datetime64(dt)) & (a >= np.datetime64(dt - pd.Timedelta(days=45)))).any():
            ret[n] = 1
    W["il_next"], W["il_fam"], W["ret"] = il_next, il_fam, ret
    D = W[(W["rec_bat_speed__n"] >= 30) & (W["ret"] == 0)].copy()
    D["z"] = (D["rec_bat_speed"] - D["prior_bat_speed"]) / (D["rec_bat_speed"] - D["prior_bat_speed"]).std()
    D["zp"], D["zm"] = D["z"].clip(lower=0), D["z"].clip(upper=0)
    res = {"label": label, "n_points": int(len(D)), "n_hitters": int(D["batter"].nunique()), "base_rate": float(D["il_next"].mean()),
           "sd_mph": float((D["rec_bat_speed"] - D["prior_bat_speed"]).std()),
           "share_forward_complete": float(D["complete"].mean()) if "complete" in D else None, "models": {}}
    for out_name, y in (("any", D["il_next"]), *[(f, (D["il_fam"] == f).astype(int)) for f in FAMS]):
        fit = sm.Logit(y, sm.add_constant(D[["z"]])).fit(disp=0, cov_type="cluster", cov_kwds={"groups": D["batter"]})
        b_, se_ = fit.params["z"], fit.bse["z"]
        res["models"][out_name] = {"OR_per_minus1SD": float(np.exp(-b_)), "lo": float(np.exp(-b_ - 1.96 * se_)),
                                   "hi": float(np.exp(-b_ + 1.96 * se_)), "p": float(fit.pvalues["z"]), "events": int(y.sum())}
        if out_name in ("any", "oblique"):          # F4e, post hoc
            f2 = sm.Logit(y, sm.add_constant(D[["zp", "zm"]])).fit(disp=0, cov_type="cluster", cov_kwds={"groups": D["batter"]})
            res["models"][out_name]["F4e"] = {
                "OR_per_plus1SD_faster": float(np.exp(f2.params["zp"])), "faster_ci": [float(np.exp(f2.conf_int().loc["zp", 0])), float(np.exp(f2.conf_int().loc["zp", 1]))],
                "faster_p": float(f2.pvalues["zp"]),
                "OR_per_1SD_slower": float(np.exp(-f2.params["zm"])), "slower_ci": [float(np.exp(-f2.conf_int().loc["zm", 1])), float(np.exp(-f2.conf_int().loc["zm", 0]))],
                "slower_p": float(f2.pvalues["zm"])}
    q = pd.qcut(D["z"].rank(method="first"), 10, labels=False)
    res["il_rate_by_decile"] = D.groupby(q)["il_next"].mean().tolist()
    return res


out["F4a_A8"] = {"complete_forward_300pa_replication": a8(False, 300, "A8 as published (forward window complete, season >= 300 PA)"),
                 "at_risk_300pa": a8(True, 300, "at risk: forward window not required, season >= 300 PA"),
                 "at_risk_no_season_requirement": a8(True, 0, "at risk: forward window not required, no season-PA requirement")}
if IL_ORIG:
    P0, _ = il_events(IL_ORIG)
    out["F4a_A8"]["original_file_complete_forward_300pa"] = a8(False, 300, "original IL file, as published", P=P0)
    out["F4a_A8"]["original_file_at_risk_no_season_requirement"] = a8(True, 0, "original IL file, at risk", P=P0)
log("A8 at-risk done", json.dumps({k: v["models"]["any"] for k, v in out["F4a_A8"].items()}, default=str)[:600])


# ------------------------------------------------------------------ F4b return to performance
POST = [(1, 5), (6, 10), (11, 20), (21, 40), (1, 10)]
hitters_by_year = {}
for (b, y) in G:
    if b in COVERED:
        hitters_by_year.setdefault(y, []).append(b)


def post_rows(b, y, S, R_, stop):
    base, src = baseline(b, y, S - pd.Timedelta(days=30))
    if base is None:
        return None
    g = G[(b, y)]
    m = (g["t"] >= np.datetime64(R_, "D")) & (g["t"] < np.datetime64(stop, "D"))
    idx = np.flatnonzero(m)[:40]
    if len(idx) == 0:
        return None
    res = {}
    for lo, hi in POST:
        k = idx[lo - 1:hi]
        if len(k) == 0:
            continue
        s = {c: g[c][k].sum() for c in g if c != "t"}
        res[f"{lo}-{hi}"] = {}
        for met in MET:
            n, dn, need = MET[met]
            if s[dn] >= need and base[dn] > 0:
                v, w = dev(s, base, met)
                res[f"{lo}-{hi}"][met] = (v, w)
    return res, src


stints = []
for p in P.itertuples():
    y = p.start.year
    if pd.isna(p.ret) or p.ret.year != y or (p.batter, y) not in G:
        continue
    later = P[(P["batter"] == p.batter) & (P["start"] > p.start)]["start"]
    stop = later.min() if len(later) else pd.Timestamp(f"{y}-12-31")
    r = post_rows(p.batter, y, p.start, p.ret, stop)
    if r is None:
        continue
    stints.append({"batter": p.batter, "fam": p.fam, "imp": p.imp, "S": p.start, "R": p.ret, "y": y, "res": r[0], "src": r[1]})
log("stints with post-return games", len(stints))
rng = np.random.default_rng(SEED + 1)
rowsB = []
for k, s in enumerate(stints):
    elig = []
    for b in hitters_by_year.get(s["y"], []):
        if b == s["batter"]:
            continue
        pl = Pb.get(b, np.array([], dtype="datetime64[ns]"))
        lo_, hi_ = np.datetime64(s["S"] - pd.Timedelta(days=60)), np.datetime64(s["R"] + pd.Timedelta(days=60))
        if ((pl >= lo_) & (pl <= hi_)).any():
            continue
        ac = Ab.get(b, np.array([], dtype="datetime64[ns]"))
        if ((ac <= np.datetime64(s["S"])) & (ac >= np.datetime64(s["S"] - pd.Timedelta(days=45)))).any():
            continue
        elig.append(b)
    rng.shuffle(elig)
    # prespecified: the first 5 usable controls in random order. Added 2026-10-06 (post hoc sensitivity, recorded in the
    # audit): every eligible control, which removes the Monte Carlo error of drawing 5 (the 5-control estimate moved by
    # up to 0.2 mph across seeds in ajsm_reconcile_return.py). The 5-control draw is unchanged by this addition.
    ctrl_all = []
    for b in elig:
        r = post_rows(b, s["y"], s["S"], s["R"], pd.Timestamp(f"{s['y']}-12-31"))
        if r is not None:
            ctrl_all.append(r[0])
    ctrl = ctrl_all[:5]
    for w_ in s["res"]:
        for met, (v, w) in s["res"][w_].items():
            cv = [c[w_][met][0] for c in ctrl if w_ in c and met in c[w_]]
            if not cv:
                continue
            cva = [c[w_][met][0] for c in ctrl_all if w_ in c and met in c[w_]]
            rowsB.append({"stint": k, "batter": s["batter"], "fam": s["fam"], "imp": s["imp"], "win": w_, "met": met, "dev": v, "w": w,
                          "ctrl_mean": float(np.mean(cv)), "n_ctrl": len(cv), "src": s["src"],
                          "ctrl_mean_all": float(np.mean(cva)), "n_ctrl_all": len(cva)})
B = pd.DataFrame(rowsB)
B["diff"] = B["dev"] - B["ctrl_mean"]
B["diff_all"] = B["dev"] - B["ctrl_mean_all"]
F4b = {"n_stints": int(B["stint"].nunique()), "by_family_n": B.drop_duplicates("stint")["fam"].value_counts().to_dict(),
       "baseline_prev_season_share": float(B.drop_duplicates("stint")["src"].eq("prev").mean()), "effects": {}}
for fam, gcol, gval in GROUPS:
    F4b["effects"][fam] = {}
    for met in MET:
        F4b["effects"][fam][met] = {}
        for w_ in [f"{lo}-{hi}" for lo, hi in POST]:
            D = B[(B["met"] == met) & (B["win"] == w_) & ((B[gcol] == gval) if gcol else True)]
            if len(D) < 5:
                continue
            e = ols_mean(D["diff"], np.ones(len(D)), D["batter"])
            e["injured_raw"] = float(D["dev"].mean()); e["control_raw"] = float(D["ctrl_mean"].mean())
            F4b["effects"][fam][met][w_] = e
            ea = ols_mean(D["diff_all"], np.ones(len(D)), D["batter"])
            ea["control_raw"] = float(D["ctrl_mean_all"].mean()); ea["controls_per_stint_median"] = float(D["n_ctrl_all"].median())
            F4b.setdefault("effects_all_controls", {}).setdefault(fam, {}).setdefault(met, {})[w_] = ea
out["F4b"] = F4b
log("F4b done", json.dumps(F4b["effects"].get("hand_wrist", {}).get("bat_speed", {}).get("1-10", {})))


# ------------------------------------------------------------------ F4c reliability and MDC (and F4d adjusted)
def window_ids(n_pa):
    """Consecutive non-overlapping windows of whole games with >= n_pa PA, within hitter-season."""
    w = np.zeros(len(bg), int) - 1
    wid = 0
    for (b, y), idx in bg.groupby(["batter", "game_year"]).indices.items():
        pa = bg["pa"].to_numpy()[idx]
        acc = 0
        cur = []
        for i, p_ in zip(idx, pa):
            cur.append(i); acc += p_
            if acc >= n_pa:
                w[cur] = wid; wid += 1; cur = []; acc = 0
    return pd.Series(w, index=bg.index)


def reliability(units, val, n_pa, wid):
    """Variance components of window means within hitter-season.

    Estimator corrected on 2026-10-06, after the first real-data run (deviation from addendum F, recorded in the audit):
    the prespecified split-half correlation of window-minus-season deviations is biased downward, because each window
    is part of the season mean it is compared with (outcome metrics came out negative). Here the noise variance of a
    window mean comes from its odd and even halves, in which any baseline cancels:
        s2_unit = (a - b)^2 / (1/n_a + 1/n_b),  noise variance of the window mean = s2_unit / (n_a + n_b);
    the total variance is the pooled within-hitter-season variance of window means (k - 1 degrees of freedom);
    reliability = 1 - noise / total; SEM = sqrt(noise); MDC95 = 1.96 * sqrt(2) * SEM (two windows) or 1.96 * SEM
    (against a stable baseline)."""
    key = bg[KEYS].assign(wid=wid.to_numpy())
    u = units.merge(key, on=KEYS, how="inner")
    u = u[u["wid"] >= 0].sort_values(["wid", "game_date", "game_pk", "at_bat_number"] + (["pitch_number"] if "pitch_number" in u else []))
    u["half"] = u.groupby("wid").cumcount() % 2
    agg = u.groupby(["wid", "half"])[val].agg(["mean", "size"]).unstack().dropna()
    a, b_ = agg[("mean", 0)], agg[("mean", 1)]
    na, nb = agg[("size", 0)].astype(float), agg[("size", 1)].astype(float)
    noise = ((a - b_) ** 2 / (1 / na + 1 / nb)) / (na + nb)
    full = u.groupby("wid")[val].mean().reindex(agg.index)
    hs = u.groupby("wid")[["batter", "game_year"]].first().reindex(agg.index)
    grp = [hs["batter"], hs["game_year"]]
    k = full.groupby(grp).transform("size")
    keep = k >= 2
    dev = (full - full.groupby(grp).transform("mean"))[keep]
    ss = (dev ** 2).groupby([hs["batter"][keep], hs["game_year"][keep]]).sum()
    dfree = full[keep].groupby([hs["batter"][keep], hs["game_year"][keep]]).size() - 1
    total = float(ss.sum() / dfree.sum())
    s2e = float(noise[keep].mean())
    rel = max(0.0, 1 - s2e / total)
    sem = float(np.sqrt(s2e))
    return {"n_windows": int(keep.sum()), "hitter_seasons": int(len(dfree)),
            "units_per_window_median": float((na + nb)[keep].median()),
            "total_sd_within_hitter_season": float(np.sqrt(total)), "noise_sd": sem,
            "true_sd": float(np.sqrt(max(0.0, total - s2e))), "reliability": rel, "sem": sem,
            "mdc95_two_windows": 1.96 * np.sqrt(2) * sem, "mdc95_vs_stable_baseline": 1.96 * sem,
            "estimator": "variance components (corrected 2026-10-06; see docstring)"}


SWr = SW.assign(bs=SW["bs_num"], sl=SW["sl_num"], fastv=SW["fast"].astype(float), bsc=SW["bsc_num"])
BIPr = BIP.assign(evv=BIP["ev_num"], hardv=BIP["hard"].astype(float))
PAr = PA.assign(xw=PA["xwoba_num"], wo=PA["woba_num"])
F4c = {}
for n_pa in (25, 50, 100):
    wid = window_ids(n_pa)
    F4c[str(n_pa)] = {"bat_speed": reliability(SWr, "bs", n_pa, wid), "swing_len": reliability(SWr, "sl", n_pa, wid),
                      "fast_swing": reliability(SWr, "fastv", n_pa, wid), "bat_speed_count_adjusted": reliability(SWr, "bsc", n_pa, wid),
                      "ev": reliability(BIPr, "evv", n_pa, wid), "hard_hit": reliability(BIPr, "hardv", n_pa, wid),
                      "xwoba": reliability(PAr, "xw", n_pa, wid), "woba": reliability(PAr, "wo", n_pa, wid)}
    log("F4c", n_pa, json.dumps({k: round(v["reliability"], 3) for k, v in F4c[str(n_pa)].items()}))
out["F4c"] = F4c

# F4d persistence of count-adjusted bat speed, 50 PA -> 50 PA, K-shuffle null (and raw bat speed on the same shuffles)
bgc = bg.copy()
bgc["bs_num"] = bgc["bsc_num"]
lgc = H.league_means(bgc)
W = C.windows(bg, 50, 50, lg)
Wc = C.windows(bgc, 50, 50, lgc)
W0 = [C.windows(C.shuffle_games(bg, SEED + 10 + k), 50, 50, lg) for k in range(K_SHUF)]
W0c = [C.windows(C.shuffle_games(bgc, SEED + 10 + k), 50, 50, lgc) for k in range(K_SHUF)]
pr = C.persistence(W, W0, ["bat_speed", "swing_len", "woba"])
prc = C.persistence(Wc, W0c, ["bat_speed"])
out["F4d"] = {"raw": pr, "count_zone_adjusted_bat_speed": prc["bat_speed"],
              "disattenuated": {m: {"persistence": pr[m]["persistence"], "reliability_50pa": F4c["50"][k]["reliability"],
                                    "disattenuated": (pr[m]["persistence"] / F4c["50"][k]["reliability"]) if F4c["50"][k]["reliability"] > 0.05 else None}
                                for m, k in (("bat_speed", "bat_speed"), ("swing_len", "swing_len"), ("woba", "woba"))}}
log("F4d", json.dumps({"raw": pr["bat_speed"]["persistence"], "adj": prc["bat_speed"]["persistence"]}))

json.dump(out, open(OUT, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
log("wrote", OUT)
