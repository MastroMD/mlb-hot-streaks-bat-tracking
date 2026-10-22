"""OJSM revision, plan addendum H (PLAN_ADDENDUM_2026-10-17h.md, written before this ran). All analyses post hoc.

python3 ojsm_new_analyses.py DATA_DIR IL_FILE CODED_CSV STATSAPI_DIR GAME_OJSM_DIR OUT_JSON [--shuffles K]

DATA_DIR       Statcast season parquet files (re-pull, see build_info.json)
IL_FILE        repo/data_public/il_transactions_hitters_2024_2026_corrected.txt
CODED_CSV      repo/data_public/il_placements_coded_2023_2026.csv
STATSAPI_DIR   transactions_2024-01-01_2026-10-05.json and schedule_2024_2026.json (MLB Stats API, public)
GAME_OJSM_DIR  folder with ojsm_f4b_stint_rows.csv and ojsm_f4b_stints.csv written by ajsm_game_ojsm.py

H1 SWC margin; H2 MDE; H3 specificity contrasts; H4 hand/wrist modifiers; H5 recovery course to game 60 with TOST;
H6 anchors (batted ball and 50-PA window); H7 re-injury after return; H8a road-only persistence; H8b pitch-mix-adjusted
persistence and reliability; H9 schedule and fatigue.
Reuses the project's definitions (common3.adjust, hot_streak_study.pitch_features/context, common3.windows/persistence).
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO = os.environ.get("HH_REPO", "/home/claude/work/Hot_Hand_Bat_Tracking/repo")
sys.path.insert(0, REPO)
import common3 as C            # noqa: E402
import hot_streak_study as H   # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
DATA, ILF, CODED, SAPI, GDIR, OUT = args[:6]
K_SHUF = int(sys.argv[sys.argv.index("--shuffles") + 1]) if "--shuffles" in sys.argv else 5
SEED = 505
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
FAM = {"O": "oblique", "H": "hamstring", "W": "hand_wrist", "X": "other"}
FAMS = ["oblique", "hamstring", "hand_wrist", "other"]
out = {"plan": "PLAN_ADDENDUM_2026-10-17h.md", "post_hoc": True}
Z90, Z95 = 1.6448536269514722, 1.959963984540054


def load_json(path):
    """MLB Stats API pulls are stored as .json or, in the shared folder, as .json.gz."""
    import gzip
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        with gzip.open(path + ".gz", "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.load(open(path))


def tost(est, se, margin=0.5):
    lo90, hi90 = est - Z90 * se, est + Z90 * se
    from scipy.stats import norm
    p = max(norm.sf((est + margin) / se), norm.sf((margin - est) / se))
    return {"est": est, "se": se, "ci90": [lo90, hi90], "tost_p": float(p), "equivalent": bool(lo90 > -margin and hi90 < margin)}


# ------------------------------------------------------------------ data (same loader rules as ajsm_game.py)
COLS = ["game_date", "game_pk", "game_type", "game_year", "at_bat_number", "pitch_number", "batter", "pitcher", "stand",
        "p_throws", "home_team", "away_team", "inning_topbot", "balls", "strikes", "zone", "description", "events", "type",
        "pitch_type", "release_speed", "launch_speed", "launch_angle", "launch_speed_angle", "estimated_woba_using_speedangle",
        "woba_value", "woba_denom", "bat_speed", "swing_length"]
files = sorted(glob.glob(os.path.join(DATA, "*.parquet")))
raw = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
raw = raw[(raw["game_type"] == "R") & raw["game_year"].between(2024, 2026)].copy()
for c in raw.columns:
    if str(raw[c].dtype) in ("Int8", "Int16", "Int32", "Int64", "Float32", "Float64"):
        raw[c] = raw[c].astype("float64")
for c in ("batter", "pitcher", "game_pk", "game_year"):
    raw[c] = raw[c].astype("int64")
raw["game_date"] = pd.to_datetime(raw["game_date"])
raw = raw.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"]).sort_values(
    ["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
log("pitches", len(raw))
d = C.adjust(raw)
d = H.pitch_features(d)
d = H.context(d)
d["road"] = (d["inning_topbot"] == "Top").astype(np.int8)        # batter's team bats in the top half = away team
d["n_pitch"] = 1

# ------------------------------------------------------------------ H6a batted-ball anchor (within hitter-season)
bb = d[(d["comp"] == 1) & (d["bip"] == 1)].copy()
bb["bs"] = bb["bs_num"]
bb["ev"] = bb["ev_num"]
bb["xwc"] = pd.to_numeric(bb["estimated_woba_using_speedangle"], errors="coerce")
bb = bb[bb["xwc"].notna()]
for c in ("bs", "ev", "xwc"):
    bb[c + "_dm"] = bb[c] - bb.groupby(["batter", "game_year"])[c].transform("mean")
H6 = {"batted_balls": int(len(bb)), "hitters": int(bb["batter"].nunique())}
for y in ("ev", "xwc"):
    f = sm.OLS(bb[y + "_dm"], bb[["bs_dm"]]).fit(cov_type="cluster", cov_kwds={"groups": bb["batter"]})
    H6[f"{y}_per_mph_bat_speed"] = {"slope": float(f.params["bs_dm"]), "se": float(f.bse["bs_dm"]),
                                    "lo": float(f.conf_int().loc["bs_dm", 0]), "hi": float(f.conf_int().loc["bs_dm", 1])}
log("H6a", json.dumps(H6))

# ------------------------------------------------------------------ batter-game panel (as ajsm_game.py) + extras
bsr = np.where(d["comp"] == 1, d["bs_num"], np.nan)
grp = [pd.to_numeric(d["balls"], errors="coerce").fillna(-1).clip(upper=3), pd.to_numeric(d["strikes"], errors="coerce").fillna(-1).clip(upper=2), d["inzone"]]
cell = pd.Series(bsr).groupby(grp).transform("mean")
d["bsc_num"] = np.where(d["comp"] == 1, bsr - cell + np.nanmean(bsr), 0.0)
# H8b pitch-mix cells: count x zone x pitch group x pitch-speed decile within group
pt = d["pitch_type"].astype(str)
pgroup = np.select([pt.isin(["FF", "SI", "FC", "FA"]), pt.isin(["SL", "ST", "CU", "KC", "SV", "CS", "SC"]),
                    pt.isin(["CH", "FS", "FO", "KN", "EP"])], ["fb", "br", "os"], "oth")
d["pgroup"] = pgroup
rs = pd.to_numeric(d["release_speed"], errors="coerce")
d["vdec"] = rs.groupby(d["pgroup"]).transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False)).fillna(-1)
cellm = pd.Series(bsr).groupby(grp + [d["pgroup"], d["vdec"]]).transform("mean")
d["bsm_num"] = np.where(d["comp"] == 1, bsr - cellm + np.nanmean(bsr), 0.0)
d["comp_road"] = (d["comp"] * d["road"]).astype(np.int8)
d["bs_road_num"] = d["bs_num"] * d["road"]
d["comp_home"] = (d["comp"] * (1 - d["road"])).astype(np.int8)
d["bs_home_num"] = d["bs_num"] * (1 - d["road"])
KEYS = ["batter", "game_year", "game_date", "game_pk"]
SUMS = C.SUM_COLS + ["bsc_num", "bsm_num", "comp_road", "bs_road_num", "comp_home", "bs_home_num", "n_pitch", "road"]
bg = d.groupby(KEYS, sort=False)[SUMS].sum().reset_index().sort_values(["batter", "game_date", "game_pk"]).reset_index(drop=True)
bg["road"] = (bg["road"] > 0).astype(int)
bg = bg[bg["pa"] > 0].reset_index(drop=True)
lg = H.league_means(bg)
SW = d.loc[d["comp"] == 1, KEYS + ["at_bat_number", "pitch_number", "bs_num", "bsm_num"]].copy()
out["inputs"] = {"n_pitches": int(len(d)), "batter_games": int(len(bg)), "statcast_files": [os.path.basename(f) for f in files]}
del d, raw
log("batter-games", len(bg))


# ------------------------------------------------------------------ H8 persistence robustness
def pers_variant(bgv, label):
    lgv = H.league_means(bgv)
    W = C.windows(bgv, 50, 50, lgv)
    W0 = [C.windows(C.shuffle_games(bgv, SEED + 10 + k), 50, 50, lgv) for k in range(K_SHUF)]
    r = C.persistence(W, W0, ["bat_speed"])["bat_speed"]
    r["label"] = label
    return r, W


H8 = {}
base_r, W_all = pers_variant(bg, "all games, competitive swings (5 shuffles; reference for the rows below)")
H8["reference_5shuffles"] = base_r
for lab, num, den in (("road_only", "bs_road_num", "comp_road"), ("home_only", "bs_home_num", "comp_home"),
                      ("pitch_mix_adjusted", "bsm_num", "comp"), ("count_zone_adjusted", "bsc_num", "comp")):
    bgv = bg.copy()
    bgv["bs_num"], bgv["comp"] = bgv[num], bgv[den]
    H8[lab], _ = pers_variant(bgv, lab)
    log("H8", lab, round(H8[lab]["persistence"], 4), H8[lab]["n"])
out["H8"] = H8


# ------------------------------------------------------------------ H6b window-level anchor (50-PA persistence windows)
def win_anchor(W):
    mnc, mnb, mnp = H.MIN_N["comp"], H.MIN_N["bip"], H.MIN_N["pa"]
    D = W[(W["season_pa"] >= 300) & (W["rec_bat_speed__n"] >= mnc) & (W["orc_bat_speed__n"] >= 3 * mnc)
          & (W["rec_ev__n"] >= mnb) & (W["orc_ev__n"] >= 3 * mnb)].copy()
    x = D["rec_bat_speed"] - D["orc_bat_speed"]
    res = {"n": int(len(D)), "hitters": int(D["batter"].nunique())}
    for m, scale in (("ev", 1.0), ("xwoba", 1000.0), ("woba", 1000.0)):
        y = (D[f"rec_{m}"] - D[f"orc_{m}"]) * scale
        f = C.ols_cl(y.to_numpy(), x.to_numpy(), D["batter"].to_numpy())
        res[m] = {"slope": float(f.params[1]), "se": float(f.bse[1]), "lo": float(f.conf_int()[1][0]), "hi": float(f.conf_int()[1][1])}
    return res


out["H6"] = {"batted_ball": H6, "window_50pa": win_anchor(W_all)}
log("H6b", json.dumps(out["H6"]["window_50pa"]))


# ------------------------------------------------------------------ reliability (ajsm_game.py estimator, verbatim logic)
def window_ids(n_pa):
    w = np.zeros(len(bg), int) - 1
    wid = 0
    for (b, y), idx in bg.groupby(["batter", "game_year"]).indices.items():
        pa = bg["pa"].to_numpy()[idx]
        acc, cur = 0, []
        for i, p_ in zip(idx, pa):
            cur.append(i); acc += p_
            if acc >= n_pa:
                w[cur] = wid; wid += 1; cur = []; acc = 0
    return pd.Series(w, index=bg.index)


def reliability(units, val, wid):
    key = bg[KEYS].assign(wid=wid.to_numpy())
    u = units.merge(key, on=KEYS, how="inner")
    u = u[u["wid"] >= 0].sort_values(["wid", "game_date", "game_pk", "at_bat_number", "pitch_number"])
    u["half"] = u.groupby("wid").cumcount() % 2
    agg = u.groupby(["wid", "half"])[val].agg(["mean", "size"]).unstack().dropna()
    a, b_ = agg[("mean", 0)], agg[("mean", 1)]
    na, nb = agg[("size", 0)].astype(float), agg[("size", 1)].astype(float)
    noise = ((a - b_) ** 2 / (1 / na + 1 / nb)) / (na + nb)
    full = u.groupby("wid")[val].mean().reindex(agg.index)
    hs = u.groupby("wid")[["batter", "game_year"]].first().reindex(agg.index)
    grp_ = [hs["batter"], hs["game_year"]]
    k = full.groupby(grp_).transform("size")
    keep = k >= 2
    dev = (full - full.groupby(grp_).transform("mean"))[keep]
    ss = (dev ** 2).groupby([hs["batter"][keep], hs["game_year"][keep]]).sum()
    dfree = full[keep].groupby([hs["batter"][keep], hs["game_year"][keep]]).size() - 1
    total = float(ss.sum() / dfree.sum())
    s2e = float(noise[keep].mean())
    sem = float(np.sqrt(s2e))
    return {"n_windows": int(keep.sum()), "reliability": max(0.0, 1 - s2e / total), "sem": sem,
            "mdc95_two_windows": 1.96 * np.sqrt(2) * sem, "mdc95_vs_stable_baseline": 1.96 * sem,
            "units_per_window_median": float((na + nb)[keep].median())}


wid50 = window_ids(50)
SWr = SW.assign(bs=SW["bs_num"], bsm=SW["bsm_num"])
out["H8"]["reliability_50pa"] = {"bat_speed": reliability(SWr, "bs", wid50), "pitch_mix_adjusted": reliability(SWr, "bsm", wid50)}
out["H6"]["window_50pa"]["bat_speed_reliability_50pa"] = out["H8"]["reliability_50pa"]["bat_speed"]["reliability"]
_r = out["H8"]["reliability_50pa"]["bat_speed"]["reliability"]
for m in ("ev", "xwoba", "woba"):
    e = out["H6"]["window_50pa"][m]
    e["disattenuated"] = {"slope": e["slope"] / _r, "lo": e["lo"] / _r, "hi": e["hi"] / _r}
log("H8 reliability", json.dumps(out["H8"]["reliability_50pa"]))

# ------------------------------------------------------------------ H9 schedule and fatigue
sch = pd.DataFrame(load_json(os.path.join(SAPI, "schedule_2024_2026.json")))
sch = sch.drop_duplicates("gamePk", keep="last").rename(columns={"gamePk": "game_pk"})
g = bg[["batter", "game_year", "game_date", "game_pk", "comp", "bs_num"]].merge(sch[["game_pk", "dayNight", "doubleHeader", "gameNumber"]], on="game_pk", how="left")
g = g.sort_values(["batter", "game_date", "game_pk"]).reset_index(drop=True)
# schedule terms from every game the hitter played (any PA), before the 5-swing filter, so that a game with few
# competitive swings neither hides the previous night game nor breaks a run of consecutive game days (revised 2026-10-17)
g["day"] = (g["dayNight"] == "day").astype(int)
prev_date = g.groupby(["batter", "game_year"])["game_date"].shift(1)
prev_night = g.groupby(["batter", "game_year"])["dayNight"].shift(1).eq("night")
g["day_after_night"] = ((g["day"] == 1) & prev_night & ((g["game_date"] - prev_date).dt.days == 1)).astype(int)
g["dh2"] = (g["doubleHeader"].isin(["Y", "S"]) & (g["gameNumber"] == 2)).astype(int)
# consecutive calendar days with a game for this hitter (doubleheaders count once)
dd = g.drop_duplicates(["batter", "game_year", "game_date"])[["batter", "game_year", "game_date"]].copy()
gap = dd.groupby(["batter", "game_year"])["game_date"].diff().dt.days.ne(1)
dd["streak"] = dd.groupby((gap).cumsum()).cumcount() + 1
g = g.merge(dd, on=["batter", "game_year", "game_date"], how="left")
g = g[g["comp"] >= 5].copy()
g["bs"] = g["bs_num"] / g["comp"]
g["dev"] = g["bs"] - (g.groupby(["batter", "game_year"])["bs_num"].transform("sum") / g.groupby(["batter", "game_year"])["comp"].transform("sum"))
g["s4_6"] = g["streak"].between(4, 6).astype(int)
g["s7_9"] = g["streak"].between(7, 9).astype(int)
g["s10"] = (g["streak"] >= 10).astype(int)
X = g[["day", "day_after_night", "dh2", "s4_6", "s7_9", "s10"]].astype(float)
# replication fix 2026-10-22: comp is int8 (pandas groupby-sum keeps the dtype), and statsmodels takes np.sqrt(weights),
# which numpy computes in float16 for int8; float weights give the exact WLS (coefficients move by < 2e-5 mph)
f = sm.WLS(g["dev"], sm.add_constant(X), weights=g["comp"].astype(float)).fit(cov_type="cluster", cov_kwds={"groups": g["batter"]})
out["H9"] = {"n_batter_games": int(len(g)), "n_with_daynight": int(g["dayNight"].notna().sum()),
             "share_day": float(g["day"].mean()), "n_day_after_night": int(g["day_after_night"].sum()), "n_dh2": int(g["dh2"].sum()),
             "streak_counts": {k: int(g[k].sum()) for k in ("s4_6", "s7_9", "s10")},
             "coef": {k: {"est": float(f.params[k]), "lo": float(f.conf_int().loc[k, 0]), "hi": float(f.conf_int().loc[k, 1]), "p": float(f.pvalues[k])}
                      for k in X.columns}}
log("H9", json.dumps(out["H9"]["coef"]))

# ------------------------------------------------------------------ stint-level analyses (H3, H4, H5, H7)
B = pd.read_csv(os.path.join(GDIR, "ojsm_f4b_stint_rows.csv"))
ST = pd.read_csv(os.path.join(GDIR, "ojsm_f4b_stints.csv"), parse_dates=["S", "R", "stop"])
CD = pd.read_csv(CODED, parse_dates=["day"]).drop_duplicates(["batter", "day"])
ST = ST.merge(CD[["batter", "day", "region", "mechanism", "impact", "text"]], left_on=["batter", "S"], right_on=["batter", "day"], how="left")
ST["days"] = (ST["R"] - ST["S"]).dt.days


def cl_mean(y, g_):
    y = np.asarray(y, float); ok = np.isfinite(y)
    f_ = sm.OLS(y[ok], np.ones(ok.sum())).fit(cov_type="cluster", cov_kwds={"groups": np.asarray(g_)[ok]})
    return {"est": float(f_.params[0]), "se": float(f_.bse[0]), "lo": float(f_.conf_int()[0][0]), "hi": float(f_.conf_int()[0][1]),
            "p": float(f_.pvalues[0]), "n": int(ok.sum())}


bs10 = B[(B["met"] == "bat_speed") & (B["win"] == "1-10")].merge(ST[["stint", "days", "mechanism", "text", "S", "R", "y"]], on="stint", how="left")
# H3 specificity contrasts
H3 = {}
for col in ("diff", "diff_all"):
    D = bs10.copy()
    Xf = pd.get_dummies(D["fam"]).astype(float)[FAMS]
    f_ = sm.OLS(D[col], Xf).fit(cov_type="cluster", cov_kwds={"groups": D["batter"]})
    con = {}
    def lc(w, name):
        t = f_.t_test(np.asarray(w, float))
        con[name] = {"est": float(np.squeeze(t.effect)), "se": float(np.squeeze(t.sd)), "lo": float(np.squeeze(t.conf_int())[0]),
                     "hi": float(np.squeeze(t.conf_int())[1]), "p": float(np.squeeze(t.pvalue))}
    lc([0, -1, 1, 0], "hand_wrist_minus_hamstring")
    lc([0, 0, 1, -1], "hand_wrist_minus_other")
    lc([-1, 1, 0, 0], "hamstring_minus_oblique")
    lc([1, -1, 0, 0], "oblique_minus_hamstring")
    lc([-1, 0, 1, 0], "hand_wrist_minus_oblique")   # added after the cold read of 2026-10-17 (the oblique deficit is of similar size)
    # hand/wrist minus all other families pooled: the stint-weighted mean of the other 3 families, as a contrast of the same
    # family model (2026-10-17 audit: the first run took it from a separate hand/wrist-vs-rest regression, kept below for the record)
    n_f = D["fam"].value_counts(); n_oth = sum(n_f[f] for f in FAMS if f != "hand_wrist")
    lc([1.0 if f == "hand_wrist" else -n_f[f] / n_oth for f in FAMS], "hand_wrist_minus_all_other")
    D2 = D.assign(hw=(D["fam"] == "hand_wrist").astype(float))
    f2 = sm.OLS(D2[col], sm.add_constant(D2[["hw"]])).fit(cov_type="cluster", cov_kwds={"groups": D2["batter"]})
    con["hand_wrist_minus_all_other_separate_regression"] = {"est": float(f2.params["hw"]), "se": float(f2.bse["hw"]), "lo": float(f2.conf_int().loc["hw", 0]),
                                                             "hi": float(f2.conf_int().loc["hw", 1]), "p": float(f2.pvalues["hw"])}
    assert abs(con["hand_wrist_minus_all_other"]["est"] - con["hand_wrist_minus_all_other_separate_regression"]["est"]) < 1e-9
    # joint test of equality across the 4 families
    R_ = np.array([[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]], float)
    w_ = f_.wald_test(R_, scalar=True)
    con["equality_of_4_families"] = {"stat": float(w_.statistic), "p": float(w_.pvalue), "df": 3}
    # impact vs non-impact
    D3 = D[D["imp"].isin(["impact", "non_impact"])].assign(im=lambda x: (x["imp"] == "impact").astype(float))
    f3 = sm.OLS(D3[col], sm.add_constant(D3[["im"]])).fit(cov_type="cluster", cov_kwds={"groups": D3["batter"]})
    con["impact_minus_non_impact"] = {"est": float(f3.params["im"]), "se": float(f3.bse["im"]), "lo": float(f3.conf_int().loc["im", 0]),
                                      "hi": float(f3.conf_int().loc["im", 1]), "p": float(f3.pvalues["im"])}
    H3[col] = {"family_means": {fm: cl_mean(D.loc[D["fam"] == fm, col], D.loc[D["fam"] == fm, "batter"]) for fm in FAMS}, "contrasts": con}
out["H3"] = H3
log("H3", json.dumps(H3["diff"]["contrasts"]["hand_wrist_minus_all_other"]))

# H4 modifiers within hand/wrist
hw = bs10[bs10["fam"] == "hand_wrist"].copy()
# rehab assignment between placement and activation
tx = load_json(os.path.join(SAPI, "transactions_2024-01-01_2026-10-05.json"))
reh = pd.DataFrame([{"batter": t["person"]["id"], "date": t.get("effectiveDate") or t.get("date")} for t in tx
                    if "rehab assignment" in (t.get("description") or "").lower() and t.get("person")])
reh["date"] = pd.to_datetime(reh["date"])


def had_rehab(b, s, r):
    x = reh[(reh["batter"] == b) & (reh["date"] >= s) & (reh["date"] <= r)]
    return int(len(x) > 0), (int((r - x["date"].min()).days) if len(x) else 0)


hw[["rehab", "rehab_days"]] = [had_rehab(b, s, r) for b, s, r in zip(hw["batter"], hw["S"], hw["R"])]
bs10_all = bs10.copy()
bs10_all[["rehab", "rehab_days"]] = [had_rehab(b, s, r) for b, s, r in zip(bs10_all["batter"], bs10_all["S"], bs10_all["R"])]
# injured side and lead/trail: stand of the hitter in games after return (season majority if needed)
stand = pd.read_parquet(files[0], columns=["batter"]).iloc[:0]
stands = []
for f in files:
    x = pd.read_parquet(f, columns=["batter", "game_year", "game_date", "stand", "woba_denom"])
    x = x[pd.to_numeric(x["woba_denom"], errors="coerce") == 1]
    stands.append(x.groupby(["batter", "game_year", "stand"]).size().rename("n").reset_index())
stands = pd.concat(stands)
sw = stands.pivot_table(index=["batter", "game_year"], columns="stand", values="n", aggfunc="sum", fill_value=0).reset_index()
sw["share_L"] = sw.get("L", 0) / (sw.get("L", 0) + sw.get("R", 0))
sw["side"] = np.where(sw["share_L"] >= 0.9, "L", np.where(sw["share_L"] <= 0.1, "R", "S"))
hw = hw.merge(sw[["batter", "game_year", "side"]], left_on=["batter", "y"], right_on=["batter", "game_year"], how="left")
txt = hw["text"].fillna("").str.lower()
hw["inj_side"] = np.where(txt.str.contains(r"\bleft\b"), "L", np.where(txt.str.contains(r"\bright\b"), "R", ""))
# lead hand: left hand for a right-handed batter (side R), right hand for a left-handed batter
hw["lead"] = np.where((hw["side"] == "R") & (hw["inj_side"] == "L"), 1, np.where((hw["side"] == "L") & (hw["inj_side"] == "R"), 1,
              np.where((hw["side"].isin(["L", "R"])) & (hw["inj_side"].isin(["L", "R"])), 0, -1)))
H4 = {"n": int(len(hw)), "median_days": float(hw["days"].median())}
med = hw["days"].median()
for col in ("diff", "diff_all"):
    r = {}
    for name, mask in (("short_stint", hw["days"] <= med), ("long_stint", hw["days"] > med),
                       ("fracture", hw["mechanism"] == "fracture"), ("not_fracture", hw["mechanism"] != "fracture"),
                       ("lead_hand", hw["lead"] == 1), ("trail_hand", hw["lead"] == 0),
                       ("rehab", hw["rehab"] == 1), ("no_rehab", hw["rehab"] == 0)):
        if mask.sum() >= 5:
            r[name] = cl_mean(hw.loc[mask, col], hw.loc[mask, "batter"])
    for name, xcol in (("per_doubling_of_stint_days", np.log2(hw["days"].clip(lower=1))),):
        f_ = sm.OLS(hw[col], sm.add_constant(pd.DataFrame({"x": xcol}))).fit(cov_type="cluster", cov_kwds={"groups": hw["batter"]})
        r[name] = {"est": float(f_.params["x"]), "lo": float(f_.conf_int().loc["x", 0]), "hi": float(f_.conf_int().loc["x", 1]), "p": float(f_.pvalues["x"])}
    for a, b_ in (("long_stint", "short_stint"), ("fracture", "not_fracture"), ("lead_hand", "trail_hand"), ("no_rehab", "rehab")):
        if a in r and b_ in r:
            m_ = hw[[col, "batter"]].assign(x=np.nan)
            sel = {"long_stint": hw["days"] > med, "short_stint": hw["days"] <= med, "fracture": hw["mechanism"] == "fracture",
                   "not_fracture": hw["mechanism"] != "fracture", "lead_hand": hw["lead"] == 1, "trail_hand": hw["lead"] == 0,
                   "rehab": hw["rehab"] == 1, "no_rehab": hw["rehab"] == 0}
            m_.loc[sel[a], "x"] = 1.0; m_.loc[sel[b_], "x"] = 0.0
            m_ = m_.dropna()
            f_ = sm.OLS(m_[col], sm.add_constant(m_[["x"]])).fit(cov_type="cluster", cov_kwds={"groups": m_["batter"]})
            r[f"{a}_minus_{b_}"] = {"est": float(f_.params["x"]), "lo": float(f_.conf_int().loc["x", 0]), "hi": float(f_.conf_int().loc["x", 1]), "p": float(f_.pvalues["x"])}
    H4[col] = r
H4["counts"] = {"fracture": int((hw["mechanism"] == "fracture").sum()), "lead": int((hw["lead"] == 1).sum()), "trail": int((hw["lead"] == 0).sum()),
                "side_unknown_or_switch": int((hw["lead"] == -1).sum()), "rehab": int(hw["rehab"].sum()),
                "rehab_days_median_among_rehab": float(hw.loc[hw["rehab"] == 1, "rehab_days"].median()) if hw["rehab"].sum() else None,
                "mechanisms": hw["mechanism"].value_counts().to_dict()}
# rehab in all families (exploratory, adjusted for family)
D = bs10_all.copy()
Xr = pd.get_dummies(D["fam"]).astype(float)[FAMS].assign(rehab=D["rehab"].astype(float))
f_ = sm.OLS(D["diff"], Xr).fit(cov_type="cluster", cov_kwds={"groups": D["batter"]})
H4["rehab_all_families_adjusted"] = {"est": float(f_.params["rehab"]), "lo": float(f_.conf_int().loc["rehab", 0]), "hi": float(f_.conf_int().loc["rehab", 1]),
                                     "p": float(f_.pvalues["rehab"]), "n_rehab": int(D["rehab"].sum()), "n": int(len(D))}
out["H4"] = H4
hw.to_csv(os.path.join(os.path.dirname(OUT), "ojsm_h4_handwrist_stints.csv"), index=False)
log("H4", json.dumps({k: H4["diff"].get(k) for k in ("long_stint_minus_short_stint", "fracture_minus_not_fracture", "lead_hand_minus_trail_hand", "no_rehab_minus_rehab")}))

# H5 recovery course with TOST (5 controls and every eligible control)
H5 = {}
for col in ("diff", "diff_all"):
    H5[col] = {}
    for fam in ["all"] + FAMS:
        H5[col][fam] = {}
        for w in ("1-5", "6-10", "11-20", "21-40", "41-60"):
            D = B[(B["met"] == "bat_speed") & (B["win"] == w) & ((B["fam"] == fam) if fam != "all" else True)]
            if len(D) < 5:
                continue
            e = cl_mean(D[col], D["batter"])
            e.update(tost(e["est"], e["se"]))
            H5[col][fam][w] = e
        # recovered: first window from which every later window is equivalent
        wins = [w for w in ("1-5", "6-10", "11-20", "21-40", "41-60") if w in H5[col][fam]]
        rec = None
        for i, w in enumerate(wins):
            if all(H5[col][fam][x]["equivalent"] for x in wins[i:]):
                rec = w; break
        H5[col][fam]["recovered_from_window"] = rec
# every tracked swing too
H5["diff_every_swing"] = {}
for fam in ["all"] + FAMS:
    H5["diff_every_swing"][fam] = {}
    for w in ("1-5", "6-10", "11-20", "21-40", "41-60"):
        D = B[(B["met"] == "bat_speed_all_swings") & (B["win"] == w) & ((B["fam"] == fam) if fam != "all" else True)]
        if len(D) < 5:
            continue
        e = cl_mean(D["diff"], D["batter"]); e.update(tost(e["est"], e["se"]))
        H5["diff_every_swing"][fam][w] = e
out["H5"] = H5
log("H5 hw", json.dumps({w: (round(v["est"], 2), v["equivalent"]) for w, v in H5["diff"]["hand_wrist"].items() if isinstance(v, dict)}))

# H7 re-injury after return: placements after the 10th game, within 30 and 60 days
Pl = []
for line in open(ILF):
    b, ev = line.strip().split(":")
    for e in ev.split():
        if e[6] == "P":
            Pl.append({"batter": int(b), "start": pd.Timestamp("20" + e[:6])})
Pl = pd.DataFrame(Pl)
season_end = bg.groupby("game_year")["game_date"].max().to_dict()
# 2026-10-17 audit: the 10th game after return is now located as in the return windows (the hitter's games in the same season
# before any later placement, a doubleheader counting as 2 games). The first run used game dates without the season restriction,
# so 13 stints whose 10th game date fell in the next season were given 1 day at risk; it is kept below for the record.
games = bg[["batter", "game_year", "game_date", "game_pk"]].sort_values(["batter", "game_date", "game_pk"])
games_first_run = bg[["batter", "game_year", "game_date"]].drop_duplicates()
same_season_rule = [True]
last_day_excluded = [0]   # stints whose 10th game was on the season's last day (counted for the text, 2026-10-17 cold read)
def h7_rows(same_season):
    same_season_rule[0] = same_season
    rows = []
    for s in ST.itertuples():
        if same_season:
            gd = games[(games["batter"] == s.batter) & (games["game_year"] == s.y) & (games["game_date"] >= s.R) & (games["game_date"] < s.stop)]["game_date"].to_numpy()
        else:
            gd = games_first_run[(games_first_run["batter"] == s.batter) & (games_first_run["game_date"] >= s.R) & (games_first_run["game_date"] < s.stop)]["game_date"].sort_values().to_numpy()
        rows += h7_row(s, gd)
    return pd.DataFrame(rows)
def h7_row(s, gd):
    if len(gd) < 10:
        return []
    t10 = pd.Timestamp(gd[9])
    x = bs10[bs10["stint"] == s.stint]
    if x.empty:
        return []
    later = Pl[(Pl["batter"] == s.batter) & (Pl["start"] > t10)]["start"]
    end_ = season_end[s.y]
    if same_season_rule[0] and (end_ - t10).days < 1:
        last_day_excluded[0] += 1
        return []      # 10th game on the season's last day: no regular-season day at risk
    return [{"stint": s.stint, "batter": s.batter, "fam": s.fam, "diff": float(x["diff"].iloc[0]), "diff_all": float(x["diff_all"].iloc[0]),
             "t10": t10, "t10_next_season": bool(t10.year != s.y),
             "inj30": int(((later > t10) & (later <= t10 + pd.Timedelta(days=30))).any()),
             "inj60": int(((later > t10) & (later <= t10 + pd.Timedelta(days=60))).any()),
             "risk30": float(min(30, max(1, (end_ - t10).days))), "risk60": float(min(60, max(1, (end_ - t10).days)))}]
def h7_fit(RI):
    H7 = {"n_stints": int(len(RI)), "events30": int(RI["inj30"].sum()), "events60": int(RI["inj60"].sum()), "n_t10_next_season": int(RI["t10_next_season"].sum())}
    for k in ("30", "60"):
        D = RI.copy()
        D["x"] = -D["diff"] / 0.5        # per 0.5 mph slower than controls
        Xf = pd.get_dummies(D["fam"]).astype(float)[FAMS].iloc[:, 1:]
        Xm = sm.add_constant(pd.concat([D[["x"]], Xf], axis=1))
        fm = sm.GLM(D[f"inj{k}"], Xm, family=sm.families.Poisson(), offset=np.log(D[f"risk{k}"])).fit(cov_type="cluster", cov_kwds={"groups": D["batter"]})
        H7[f"rr_per_0p5mph_slower_{k}d"] = {"est": float(np.exp(fm.params["x"])), "lo": float(np.exp(fm.conf_int().loc["x", 0])),
                                            "hi": float(np.exp(fm.conf_int().loc["x", 1])), "p": float(fm.pvalues["x"])}
    return H7
RI = h7_rows(True); H7 = h7_fit(RI); assert H7["n_t10_next_season"] == 0
H7["n_excluded_10th_game_on_last_day"] = last_day_excluded[0]
out["H7"] = H7
out["H7_first_run_definition"] = h7_fit(h7_rows(False))
log("H7", json.dumps(H7)); log("H7 first-run definition", json.dumps(out["H7_first_run_definition"]))

json.dump(out, open(OUT, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
log("wrote", OUT)
