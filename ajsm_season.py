"""AJSM analyses F1-F3 (plan addendum F, PLAN_ADDENDUM_2026-10-05f.md, written before this ran).

python3 ajsm_season.py IL_FILE FG_BATTRACK_CSV FG_POSITION_XLSX FG_2026_CSV OUT_JSON [CODED_CSV]

CODED_CSV (addendum G): ajsm_il_coding.py output; adds F3 by body region and mechanism. Run once on the original IL file
(prespecified) and once on the corrected file with CODED_CSV (addendum G).

F1  season-level bat-tracking risk factors for IL placement (logistic, hitter-clustered)
F2  change in swing metrics from the season before to the season after an IL season, by family (OLS, clustered)
F3  IL epidemiology in the cohort (descriptive)

Deviation from the plan, recorded in the output: the FanGraphs position-player file has no position letter (its 'Pos'
column is the positional adjustment in runs), so the prespecified catcher covariate cannot be built. It is dropped, and a
sensitivity adds the positional adjustment per 600 PA as a continuous proxy for defensive demand.
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

IL, BT, XL, F26, OUT = sys.argv[1:6]
CODED = sys.argv[6] if len(sys.argv) > 6 else None
FAM = {"O": "oblique", "H": "hamstring", "W": "hand_wrist", "X": "other"}
sha16 = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


# ------------------------------------------------------------------ inputs
def il_events():
    P, A = [], []
    for line in open(IL):
        b, ev = line.strip().split(":")
        for e in ev.split():
            d = pd.Timestamp("20" + e[:6])
            if e[6] == "P":
                P.append({"batter": int(b), "start": d, "fam": FAM[e[7]], "il_days": e[8:]})
            else:
                A.append({"batter": int(b), "end": d})
    P = pd.DataFrame(P).sort_values(["batter", "start"]).reset_index(drop=True)
    A = pd.DataFrame(A).sort_values(["batter", "end"]).reset_index(drop=True)
    P["year"] = P["start"].dt.year
    # next activation after each placement, before the hitter's next placement
    act, nxt = [], []
    for b, g in P.groupby("batter", sort=False):
        a = A.loc[A["batter"] == b, "end"].tolist()
        st = g["start"].tolist()
        for k, s in enumerate(st):
            lim = st[k + 1] if k + 1 < len(st) else pd.Timestamp("2100-01-01")
            cand = [x for x in a if s <= x <= lim]
            act.append(cand[0] if cand else pd.NaT)
    P["activation"] = act
    season_end = pd.to_datetime(P["year"].astype(str) + "-10-01")
    P["season_ending"] = P["activation"].isna() | (P["activation"] >= season_end)
    P["stint_days"] = (P["activation"] - P["start"]).dt.days
    return P, A


def fangraphs():
    X = pd.read_excel(XL)
    X = X[X["Season"].between(2021, 2025)].dropna(subset=["MLBAMID"]).copy()
    X["MLBAMID"] = X["MLBAMID"].astype(int)
    X = X[["Season", "MLBAMID", "PA", "Age", "Pos", "xwOBA", "wRC+", "SwgLng"]]
    Z = pd.read_csv(F26).dropna(subset=["MLBAMID"])
    Z["MLBAMID"] = Z["MLBAMID"].astype(int)
    Z = Z[["Season", "MLBAMID", "PA", "Age", "xwOBA", "wRC+"]]
    S = pd.concat([X, Z], ignore_index=True)
    B = pd.read_csv(BT).dropna(subset=["MLBAMID"])
    B["MLBAMID"] = B["MLBAMID"].astype(int)
    B = B[B["Season"] >= 2023][["Season", "MLBAMID", "BatSpd", "FastSw%", "CompSw", "Tilt", "AtkAng", "AtkDir"]]
    S = S.merge(B, on=["Season", "MLBAMID"], how="outer")
    S = S.rename(columns={"MLBAMID": "batter", "Season": "season", "FastSw%": "fast", "wRC+": "wrc"})
    assert not S.duplicated(["batter", "season"]).any()
    return S


P, A = il_events()
S = fangraphs()
W = S.set_index(["batter", "season"])
maxpa = S[S["season"].between(2024, 2026)].groupby("batter")["PA"].max()
U = set(maxpa.index[maxpa >= 330])
AMBIG = set(maxpa.index[(maxpa >= 300) & (maxpa < 330)])
il_hitters = set(P["batter"])
out = {"plan": {"file": "PLAN_ADDENDUM_2026-10-05f.md", "sha16": "b21ecb192b5903d0", "addendum_g": "PLAN_ADDENDUM_2026-10-05g.md", "addendum_g_sha16": "95404f233ccc6d1a"},
       "inputs": {k: sha16(p) for k, p in (("il", IL), ("fg_battrack", BT), ("fg_position_xlsx", XL), ("fg_2026", F26))},
       "universe": {"hitters_maxPA_ge330": len(U), "hitters_maxPA_300_329_excluded": len(AMBIG),
                    "il_hitters": len(il_hitters), "il_hitters_in_U": len(il_hitters & U),
                    "il_hitters_maxPA_lt300": int(sum(maxpa.reindex(list(il_hitters)) < 300))},
       "deviations": ["Catcher covariate not built: the FanGraphs file has no position letter ('Pos' is the positional "
                      "adjustment in runs). Sensitivity adds positional adjustment per 600 PA instead."]}


def get(b, y, c):
    try:
        return W.at[(b, y), c]
    except KeyError:
        return np.nan


def il_in(b, y, fam=None):
    g = P[(P["batter"] == b) & (P["year"] == y)]
    if fam is not None:
        g = g[g["fam"] == fam]
    return int(len(g) > 0)


def first_fam(b, y):
    g = P[(P["batter"] == b) & (P["year"] == y)].sort_values("start")
    return g["fam"].iloc[0] if len(g) else "none"


def or_row(fit, term, scale=1.0):
    b, se = fit.params[term] * scale, fit.bse[term] * scale
    return {"OR": float(np.exp(b)), "lo": float(np.exp(b - 1.96 * se)), "hi": float(np.exp(b + 1.96 * se)),
            "p": float(fit.pvalues[term])}


def logit(df, y, terms):
    f = smf.logit(f"{y} ~ " + " + ".join(terms), data=df).fit(disp=0, cov_type="cluster",
                                                                   cov_kwds={"groups": df["batter"]})
    return f


# ------------------------------------------------------------------ F1
rows = []
for Y in (2024, 2025, 2026):
    for b in S.loc[S["season"] == Y - 1, "batter"].unique():
        pa1, cs1 = get(b, Y - 1, "PA"), get(b, Y - 1, "CompSw")
        played = (get(b, Y, "PA") or 0) > 0 or il_in(b, Y)
        rows.append({"batter": b, "Y": Y, "pa_prev": pa1, "comp_prev": cs1, "played": played,
                     "inU": b in U, "bs": get(b, Y - 1, "BatSpd"), "sl": get(b, Y - 1, "SwgLng"),
                     "fast": get(b, Y - 1, "fast"), "aa": get(b, Y - 1, "AtkAng"), "ad": get(b, Y - 1, "AtkDir"),
                     "tilt": get(b, Y - 1, "Tilt"), "bs2": get(b, Y - 2, "BatSpd"), "comp2": get(b, Y - 2, "CompSw"),
                     "age": get(b, Y - 1, "Age") + 1, "posadj600": get(b, Y - 1, "Pos") / pa1 * 600 if pa1 else np.nan,
                     "il_prev": il_in(b, Y - 1) if Y - 1 >= 2024 else np.nan,
                     "y_any": il_in(b, Y), "y_obl": il_in(b, Y, "oblique"), "y_ham": il_in(b, Y, "hamstring"),
                     "y_hw": il_in(b, Y, "hand_wrist"), "y_oth": il_in(b, Y, "other")})
R = pd.DataFrame(rows)
R["dbs"] = R["bs"] - R["bs2"]
R.loc[~(R["comp2"] >= 100), "dbs"] = np.nan
R["pa100"] = R["pa_prev"] / 100
R["sl05"] = R["sl"] / 0.5
R["fast10"] = R["fast"] * 10
R["aa5"], R["ad5"], R["tilt5"] = R["aa"] / 5, R["ad"] / 5, R["tilt"] / 5
R["season"] = R["Y"].astype(str)
prim = R[R["Y"].isin([2025, 2026]) & (R["pa_prev"] >= 330) & (R["comp_prev"] >= 200) & R["played"] & R["bs"].notna()].copy()
assert prim["batter"].isin(U).all()
COV = ["age", "pa100", "il_prev", "C(season)"]
F1 = {"cohort": {"hitter_seasons": int(len(prim)), "hitters": int(prim["batter"].nunique()),
                 "events": {k: int(prim[k].sum()) for k in ("y_any", "y_obl", "y_ham", "y_hw", "y_oth")},
                 "bs_mean_sd": [float(prim["bs"].mean()), float(prim["bs"].std())],
                 "age_mean_sd": [float(prim["age"].mean()), float(prim["age"].std())],
                 "pa_prev_median": float(prim["pa_prev"].median()), "il_prev_share": float(prim["il_prev"].mean())},
      "primary": {}, "secondary_exposures": {}, "sensitivity": {}}
for y in ("y_obl", "y_any", "y_ham", "y_hw", "y_oth"):
    f = logit(prim, y, ["bs"] + COV)
    F1["primary"][y] = {"bat_speed_per_mph": or_row(f, "bs"), "age_per_year": or_row(f, "age"),
                        "pa_per_100": or_row(f, "pa100"), "prior_il": or_row(f, "il_prev"), "n": int(f.nobs)}
for ex in ("sl05", "fast10", "aa5", "ad5", "tilt5", "dbs"):
    F1["secondary_exposures"][ex] = {}
    d = prim[prim[ex].notna()]
    for y in ("y_obl", "y_any", "y_ham", "y_hw"):
        f = logit(d, y, [ex] + COV)
        F1["secondary_exposures"][ex][y] = dict(or_row(f, ex), n=int(f.nobs), events=int(d[y].sum()))
# sensitivity (a): add 2024 with 2023 exposures (no prior-IL covariate)
sa = R[(R["pa_prev"] >= 330) & R["played"] & R["bs"].notna() & R["inU"] &
       (((R["Y"] >= 2025) & (R["comp_prev"] >= 200)) | ((R["Y"] == 2024) & (R["comp_prev"] >= 100)))].copy()
# (b) z within season; (c) no prior-IL covariate; (d) positional adjustment proxy
prim["bsz"] = prim.groupby("Y")["bs"].transform(lambda s: (s - s.mean()) / s.std())
for y in ("y_obl", "y_any", "y_ham", "y_hw"):
    F1["sensitivity"][y] = {
        "a_add_2024": dict(or_row(logit(sa, y, ["bs", "age", "pa100", "C(season)"]), "bs"), n=int(len(sa)), events=int(sa[y].sum())),
        "b_z_within_season": or_row(logit(prim, y, ["bsz"] + COV), "bsz"),
        "c_no_prior_il": or_row(logit(prim, y, ["bs", "age", "pa100", "C(season)"]), "bs"),
        "d_posadj_proxy": or_row(logit(prim[prim["posadj600"].notna()], y, ["bs", "posadj600"] + COV), "bs")}
# descriptive: IL rate by bat-speed quintile
prim["q"] = pd.qcut(prim["bs"], 5, labels=False)
F1["by_quintile"] = prim.groupby("q").agg(bs=("bs", "mean"), n=("bs", "size"), any=("y_any", "mean"),
                                          obl=("y_obl", "mean"), ham=("y_ham", "mean"), hw=("y_hw", "mean")).reset_index().to_dict(orient="list")
out["F1"] = F1

# ------------------------------------------------------------------ F2
rows = []
for Y in (2024, 2025):
    for b in U:
        b0, b1, c0, c1 = get(b, Y - 1, "BatSpd"), get(b, Y + 1, "BatSpd"), get(b, Y - 1, "CompSw"), get(b, Y + 1, "CompSw")
        if not (c0 >= 100 and c1 >= 100) or np.isnan(b0) or np.isnan(b1):
            continue
        fams = set(P.loc[(P["batter"] == b) & (P["year"] == Y), "fam"])
        rows.append({"batter": b, "Y": Y, "fam": first_fam(b, Y), "any_hw": int("hand_wrist" in fams),
                     "il_next": il_in(b, Y + 1), "age": get(b, Y, "Age") if not np.isnan(get(b, Y, "Age")) else get(b, Y - 1, "Age") + 1,
                     "bs0": b0, "bs1": b1, "bsY": get(b, Y, "BatSpd"), "cY": get(b, Y, "CompSw"), "c0": c0, "c1": c1,
                     "f0": get(b, Y - 1, "fast"), "f1": get(b, Y + 1, "fast"),
                     "x0": get(b, Y - 1, "xwOBA"), "x1": get(b, Y + 1, "xwOBA"),
                     "pa0": get(b, Y - 1, "PA"), "pa1": get(b, Y + 1, "PA")})
D = pd.DataFrame(rows)
D["d_bs"] = D["bs1"] - D["bs0"]
D["d_bs_inj"] = np.where(D["cY"] >= 100, D["bsY"] - D["bs0"], np.nan)
D["d_fast"] = (D["f1"] - D["f0"]) * 100
D["d_xw"] = (D["x1"] - D["x0"]) * 1000
D["age2"] = (D["age"] - 28) ** 2
D["season"] = D["Y"].astype(str)
D["fam"] = pd.Categorical(D["fam"], categories=["none", "oblique", "hamstring", "hand_wrist", "other"])
D["w"] = 2 / (1 / D["c0"] + 1 / D["c1"])


def ols(df, y, base, extra=(), weights=None, fam_col="fam"):
    df = df[df[y].notna() & df[base].notna()]
    terms = [fam_col, "age", "age2", "C(season)", base] + list(extra)
    form = f"{y} ~ " + " + ".join(terms)
    m = smf.wls(form, data=df, weights=df[weights] if weights else np.ones(len(df)))
    f = m.fit(cov_type="cluster", cov_kwds={"groups": df["batter"]})
    res = {"n": int(f.nobs)}
    for t in f.params.index:
        if t.startswith(fam_col):
            k = t.split("T.")[-1].rstrip("]") if "T." in t else t
            res[k] = {"b": float(f.params[t]), "lo": float(f.conf_int().loc[t, 0]), "hi": float(f.conf_int().loc[t, 1]),
                      "p": float(f.pvalues[t]), "n": int((df[fam_col] == k).sum()) if fam_col == "fam" else int(df[fam_col].sum())}
    res["age"] = {"b": float(f.params["age"]), "lo": float(f.conf_int().loc["age", 0]), "hi": float(f.conf_int().loc["age", 1])}
    return res


F2 = {"cohort": {"hitter_seasons": int(len(D)), "hitters": int(D["batter"].nunique()),
                 "by_family": D["fam"].value_counts().reindex(D["fam"].cat.categories).astype(int).to_dict()},
      "raw_mean_change": D.groupby("fam", observed=False)["d_bs"].agg(["mean", "std", "size"]).reset_index().to_dict(orient="list"),
      "primary_d_bs": ols(D, "d_bs", "bs0"),
      "secondary": {"d_bs_injury_season": ols(D.assign(base=D["bs0"]), "d_bs_inj", "bs0"),
                    "d_fast_pp": ols(D.assign(f0p=D["f0"] * 100), "d_fast", "f0p"),
                    "d_xwoba_pts": ols(D.assign(x0p=D["x0"] * 1000), "d_xw", "x0p")},
      "sensitivity": {"a_exclude_il_next": ols(D[D["il_next"] == 0], "d_bs", "bs0"),
                      "b_weighted": ols(D, "d_bs", "bs0", weights="w")},
      # exploratory, added after the independent check: 2023 bat tracking covers only the second half of that season,
      # so index season 2024 has a half-season baseline; index 2025 alone uses full seasons on both sides
      "exploratory_index_2025_only": ols(D[D["Y"] == 2025], "d_bs", "bs0")}
D["hw"] = D["any_hw"]
f = smf.ols("d_bs ~ hw + age + age2 + C(season) + bs0", data=D).fit(cov_type="cluster", cov_kwds={"groups": D["batter"]})
F2["sensitivity"]["c_any_hand_wrist"] = {"b": float(f.params["hw"]), "lo": float(f.conf_int().loc["hw", 0]),
                                         "hi": float(f.conf_int().loc["hw", 1]), "p": float(f.pvalues["hw"]), "n_hw": int(D["hw"].sum())}
out["F2"] = F2

# ------------------------------------------------------------------ F3
PU = P[P["batter"].isin(U)].copy()
pa_U = S[S["batter"].isin(U) & S["season"].between(2024, 2026)].groupby("season")["PA"].sum()
F3 = {"placements": int(len(PU)), "hitters_with_placement": int(PU["batter"].nunique()), "cohort_hitters": len(U),
      "by_season_family": PU.groupby(["year", "fam"]).size().unstack(fill_value=0).to_dict(orient="index"),
      "by_family": PU["fam"].value_counts().to_dict(),
      "per_1000_PA_by_season": {int(y): float(1000 * (PU["year"] == y).sum() / pa_U[y]) for y in (2024, 2025, 2026)},
      "il_list_type": PU["il_days"].value_counts().to_dict(),
      "stint_days_by_family": {}, "season_ending_share_by_family": {}, "repeat_same_family_365d": {}}
for fam, g in PU.groupby("fam"):
    s = g.loc[~g["season_ending"], "stint_days"]
    F3["stint_days_by_family"][fam] = {"median": float(s.median()), "q1": float(s.quantile(.25)), "q3": float(s.quantile(.75)), "n": int(len(s))}
    F3["season_ending_share_by_family"][fam] = float(g["season_ending"].mean())
    rep = 0
    for b, h in g.groupby("batter"):
        st = h["start"].sort_values().tolist()
        rep += sum((st[k] - st[k - 1]).days <= 365 for k in range(1, len(st)))
    F3["repeat_same_family_365d"][fam] = {"repeats": int(rep), "placements": int(len(g))}
s = PU.loc[~PU["season_ending"], "stint_days"]
F3["stint_days_all"] = {"median": float(s.median()), "q1": float(s.quantile(.25)), "q3": float(s.quantile(.75)), "n": int(len(s))}
F3["season_ending_share_all"] = float(PU["season_ending"].mean())
if CODED:
    cd = pd.read_csv(CODED, parse_dates=["day"])
    PU = PU.merge(cd[["batter", "day", "region", "mechanism", "impact"]].drop_duplicates(["batter", "day"]),
                  left_on=["batter", "start"], right_on=["batter", "day"], how="left")
    F3["coded_match_share"] = float(PU["region"].notna().mean())
    for col in ("region", "mechanism", "impact"):
        tab = {}
        for k, g in PU.groupby(col):
            s_ = g.loc[~g["season_ending"], "stint_days"]
            tab[str(k)] = {"placements": int(len(g)), "hitters": int(g["batter"].nunique()),
                           "share": float(len(g) / len(PU)), "stint_median": float(s_.median()) if len(s_) else None,
                           "stint_q1": float(s_.quantile(.25)) if len(s_) else None, "stint_q3": float(s_.quantile(.75)) if len(s_) else None,
                           "season_ending_share": float(g["season_ending"].mean())}
        F3[f"by_{col}"] = tab
    F3["region_by_mechanism"] = pd.crosstab(PU["region"], PU["mechanism"]).to_dict(orient="index")
out["F3"] = F3

json.dump(out, open(OUT, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps({"F1_obl_bs": F1["primary"]["y_obl"]["bat_speed_per_mph"], "F1_any_bs": F1["primary"]["y_any"]["bat_speed_per_mph"],
                  "F2_hw": F2["primary_d_bs"].get("hand_wrist"), "F1_n": F1["cohort"], "F2_n": F2["cohort"]}, indent=1, default=str))
