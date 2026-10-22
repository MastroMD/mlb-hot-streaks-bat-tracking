"""Reconcile the AJSM season-level oblique result (F1, H1) with the companion Paper 1 (Bat-Tracking + Injuries).

Post hoc; not prespecified. Paper 1 (2026-10-06 build) reports that prior-season bat speed predicts next-season oblique
strain (lagged GEE Poisson, IRR 1.47 per SD, 1,320 pairs, 93 strains); F1 here reports OR 1.03 per mph (about 1.08
per SD) in 504 hitter-seasons with 40 oblique placements. This script starts from Paper 1's own lagged panel and moves
one design element at a time to the F1 design, then the reverse, and compares outcome labels and exposures for the same
hitter-seasons.

python3 reconcile_paper1.py P1_DIR IL_ORIG IL_CORR FG_BATTRACK FG_XLSX FG_2026 OUT_JSON
P1_DIR holds data/audit_2026-10-06/{panel,il_stints_all}.csv and output/audit_2026-10-06/review_lag_panel.csv.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable, Independence
from statsmodels.genmod.families import Binomial, Poisson

warnings.filterwarnings("ignore")
P1, IL_ORIG, IL_CORR, BT, XL, F26, OUT = sys.argv[1:8]
SD = 2.68                                     # Paper 1's bat-speed SD, used for every per-SD estimate here
ADJ_P1 = "age + bmi + C(pos_group) + switch + lefty + C(season_f)"
ADJ_F1 = "age_f1 + pa100 + il_prev + C(season_f)"


# ------------------------------------------------------------------ inputs
def il_file(path):
    rows = []
    for line in open(path):
        b, ev = line.strip().split(":")
        for e in ev.split():
            if e[6] == "P":
                rows.append({"batter": int(b), "start": pd.Timestamp("20" + e[:6]), "fam": e[7]})
    d = pd.DataFrame(rows)
    d["year"] = d["start"].dt.year
    return d


def fangraphs():
    X = pd.read_excel(XL)
    X = X[X["Season"].between(2021, 2025)].dropna(subset=["MLBAMID"]).copy()
    X["MLBAMID"] = X["MLBAMID"].astype(int)
    X = X[["Season", "MLBAMID", "PA", "Age"]]
    Z = pd.read_csv(F26).dropna(subset=["MLBAMID"])
    Z["MLBAMID"] = Z["MLBAMID"].astype(int)
    S = pd.concat([X, Z[["Season", "MLBAMID", "PA", "Age"]]], ignore_index=True)
    B = pd.read_csv(BT).dropna(subset=["MLBAMID"])
    B["MLBAMID"] = B["MLBAMID"].astype(int)
    B = B[B["Season"] >= 2023][["Season", "MLBAMID", "BatSpd", "CompSw"]]
    S = S.merge(B, on=["Season", "MLBAMID"], how="outer").rename(columns={"MLBAMID": "mlbam", "Season": "season"})
    return S.groupby(["mlbam", "season"], as_index=False).first()


L = pd.read_csv(os.path.join(P1, "output/audit_2026-10-06/review_lag_panel.csv"))
L["pos_group"] = pd.Categorical(L["pos_group"], categories=["C", "CIF", "MIF", "OF", "DH"])
L["season_f"] = L["season"].astype(str)
ST = pd.read_csv(os.path.join(P1, "data/audit_2026-10-06/il_stints_all.csv"), parse_dates=["start", "placed"])
FG = fangraphs()
IO, IC = il_file(IL_ORIG), il_file(IL_CORR)

# F1 variables for each lagged pair (T = season, Y = ns)
fgT = FG.rename(columns={"season": "season", "PA": "pa_prev", "CompSw": "comp_prev", "BatSpd": "bs_fg", "Age": "age_fgT"})
L = L.merge(fgT[["mlbam", "season", "pa_prev", "comp_prev", "bs_fg", "age_fgT"]], on=["mlbam", "season"], how="left")
fgY = FG.rename(columns={"season": "ns", "PA": "pa_Y"})[["mlbam", "ns", "pa_Y"]]
L = L.merge(fgY, on=["mlbam", "ns"], how="left")
for tag, I in (("orig", IO), ("corr", IC)):
    obl = I[I["fam"] == "O"].groupby(["batter", "year"]).size().rename(f"y_obl_{tag}").reset_index()
    anyp = I.groupby(["batter", "year"]).size().rename(f"il_{tag}").reset_index()
    L = L.merge(obl.rename(columns={"batter": "mlbam", "year": "ns"}), on=["mlbam", "ns"], how="left")
    L = L.merge(anyp.rename(columns={"batter": "mlbam", "year": "season", f"il_{tag}": f"il_prev_{tag}"}), on=["mlbam", "season"], how="left")
    L = L.merge(anyp.rename(columns={"batter": "mlbam", "year": "ns", f"il_{tag}": f"il_Y_{tag}"}), on=["mlbam", "ns"], how="left")
    for c in (f"y_obl_{tag}", f"il_prev_{tag}", f"il_Y_{tag}"):
        L[c] = (L[c].fillna(0) > 0).astype(int)
L["age_f1"] = L["age_fgT"] + 1
L["pa100"] = L["pa_prev"] / 100
L["il_prev"] = L["il_prev_orig"]
# IL-file universe: hitters with a >=330-PA FanGraphs season in 2024-2026 (F1's rule; all F1 hitters satisfy it)
maxpa = FG[FG["season"].between(2024, 2026)].groupby("mlbam")["PA"].max()
L["inU"] = L["mlbam"].map(maxpa).fillna(0) >= 330
L["played"] = (L["pa_Y"].fillna(0) > 0) | (L["il_Y_orig"] > 0)
L["f1_elig"] = (L["pa_prev"] >= 330) & (L["comp_prev"] >= 200) & L["played"] & L["bs_fg"].notna()
L["p1_any_ob"] = (L["nx_ob"] > 0).astype(int)

# Paper 1 oblique placements with both dates, for label comparison
OB = ST[ST["site_fine"] == "oblique_strict"].copy()
OB["year_start"], OB["year_placed"] = OB["start"].dt.year, OB["placed"].dt.year


# ------------------------------------------------------------------ models
def fit(d, y, x="avg_bat_speed", adj=ADJ_P1, kind="poisson_offset", scale=SD):
    d = d.sort_values(["mlbam", "season"]).copy()
    f = f"{y} ~ {x} + {adj}"
    if kind == "poisson_offset":
        r = smf.gee(f, groups="mlbam", data=d, family=Poisson(), cov_struct=Exchangeable(), offset=d["nx_logoff"]).fit()
    elif kind == "gee_logit":
        r = smf.gee(f, groups="mlbam", data=d, family=Binomial(), cov_struct=Exchangeable()).fit()
    elif kind == "logit_cluster":
        r = smf.logit(f, data=d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d["mlbam"]})
    b, se = r.params[x] * scale, r.bse[x] * scale
    return {"per_SD": float(np.exp(b)), "lo": float(np.exp(b - 1.96 * se)), "hi": float(np.exp(b + 1.96 * se)),
            "p": float(r.pvalues[x]), "per_mph": float(np.exp(r.params[x])), "n": int(len(d)), "events": int((d[y] > 0).sum()),
            "logb_per_SD": float(b), "se_per_SD": float(se)}


res = {"note": "post hoc reconciliation with Paper 1; every per-SD estimate uses SD = 2.68 mph (Paper 1)", "steps": {}, "reverse": {}}
S = res["steps"]
S["0 Paper 1 lagged as published (all pairs, count, Paper 1 covariates, offset)"] = fit(L, "nx_ob")
L25 = L[L["ns"] >= 2025]
S["1 index seasons 2025-2026 only"] = fit(L25, "nx_ob")
E = L25[L25["f1_elig"]]
S["2 + F1 eligibility (prior-season PA >= 330, >= 200 competitive swings, played)"] = fit(E, "nx_ob")
S["3 + any oblique placement (binary, GEE logistic)"] = fit(E, "p1_any_ob", kind="gee_logit")
S["4 + F1 covariates (age, prior PA, prior IL, season)"] = fit(E, "p1_any_ob", adj=ADJ_F1, kind="gee_logit")
S["5 + F1 outcome labels (IL file, record-date dating, oblique family incl. intercostal)"] = fit(E, "y_obl_orig", adj=ADJ_F1, kind="gee_logit")
S["6 + F1 exposure (FanGraphs season bat speed)"] = fit(E, "y_obl_orig", x="bs_fg", adj=ADJ_F1, kind="gee_logit")
S["7 + F1 model (logistic, cluster-robust)"] = fit(E, "y_obl_orig", x="bs_fg", adj=ADJ_F1, kind="logit_cluster")
S["7b same with corrected IL dates"] = fit(E, "y_obl_corr", x="bs_fg", adj=ADJ_F1, kind="logit_cluster")

# reverse: from the F1 design, add one Paper 1 element at a time
Rv = res["reverse"]
Rv["0 F1 design on the overlap (labels: IL file original)"] = S["7 + F1 model (logistic, cluster-robust)"]
Rv["1 Paper 1 labels"] = fit(E, "p1_any_ob", x="bs_fg", adj=ADJ_F1, kind="logit_cluster")
Rv["2 F1 labels, Paper 1 covariates"] = fit(E, "y_obl_orig", x="bs_fg", adj=ADJ_P1, kind="logit_cluster")
L_U = L[L["ns"] >= 2025]
L_U = L_U[L_U["inU"] & L_U["comp_prev"].notna()]
Rv["3 F1 labels, index 2025-2026, all covered hitters with >= 100 competitive swings (no PA rule)"] = fit(L_U, "y_obl_orig", x="avg_bat_speed", adj=ADJ_F1.replace("pa100 + ", "").replace("il_prev + ", "il_prev + "), kind="logit_cluster")
Lall_U = L[L["inU"]]
Rv["4 F1 labels (corrected), all index seasons 2024-2026, covered hitters, Paper 1 covariates, Poisson"] = fit(
    Lall_U.assign(y=Lall_U["y_obl_corr"]), "y", kind="poisson_offset")
Rv["5 Paper 1 labels on the same covered hitters (all index seasons)"] = fit(Lall_U, "nx_ob")
# part-time hitters outside the IL file (383 pairs, 11 strains): the Poisson GEE does not converge; logistic reported
try:
    Rv["6 Paper 1 labels, hitters NOT covered by the IL file (part-time; logistic)"] = fit(L[~L["inU"]], "p1_any_ob", kind="logit_cluster")
except Exception as e:  # noqa: BLE001
    res["reverse_6_error"] = str(e)

# ------------------------------------------------------------------ outcome labels for the same hitter-seasons
lab = {}
for name, d in (("F1-eligible pairs, index 2025-2026", E), ("covered hitters, all index seasons", Lall_U)):
    ct = pd.crosstab(d["p1_any_ob"], d["y_obl_orig"])
    ctc = pd.crosstab(d["p1_any_ob"], d["y_obl_corr"])
    lab[name] = {"n": int(len(d)), "paper1_yes": int(d["p1_any_ob"].sum()), "f1_orig_yes": int(d["y_obl_orig"].sum()),
                 "f1_corr_yes": int(d["y_obl_corr"].sum()),
                 "both_orig": int(((d["p1_any_ob"] == 1) & (d["y_obl_orig"] == 1)).sum()),
                 "paper1_only_orig": int(((d["p1_any_ob"] == 1) & (d["y_obl_orig"] == 0)).sum()),
                 "f1_only_orig": int(((d["p1_any_ob"] == 0) & (d["y_obl_orig"] == 1)).sum()),
                 "both_corr": int(((d["p1_any_ob"] == 1) & (d["y_obl_corr"] == 1)).sum()),
                 "paper1_only_corr": int(((d["p1_any_ob"] == 1) & (d["y_obl_corr"] == 0)).sum()),
                 "f1_only_corr": int(((d["p1_any_ob"] == 0) & (d["y_obl_corr"] == 1)).sum())}
res["labels"] = lab
# list the discordant hitter-seasons with both sources' text
disc = E[(E["p1_any_ob"] != E["y_obl_corr"])][["mlbam", "name", "season", "ns", "avg_bat_speed", "bs_fg", "nx_ob", "y_obl_orig", "y_obl_corr"]]
rows = []
for r in disc.itertuples():
    p1 = OB[(OB["mlbam"] == r.mlbam) & (OB["season"] == r.ns)][["start", "placed", "diagnosis"]].astype(str).values.tolist()
    ours = IC[(IC["batter"] == r.mlbam) & (IC["year"] == r.ns)][["start", "fam"]].astype(str).values.tolist()
    allp1 = ST[(ST["mlbam"] == r.mlbam) & (ST["season"] == r.ns)][["start", "site_fine", "diagnosis"]].astype(str).values.tolist()
    rows.append({"mlbam": r.mlbam, "name": r.name, "Y": int(r.ns), "bs_T": round(r.avg_bat_speed, 2), "paper1_oblique": int(r.nx_ob),
                 "f1_oblique_corr": int(r.y_obl_corr), "paper1_oblique_rows": p1, "il_file_rows_Y": ours, "paper1_all_rows_Y": allp1})
res["discordant_F1_eligible"] = rows

# ------------------------------------------------------------------ exposures for the same hitter-seasons
ex = E[["avg_bat_speed", "bs_fg"]].dropna()
res["exposure"] = {"n": int(len(ex)), "r": float(ex.corr().iloc[0, 1]), "mean_diff": float((ex["avg_bat_speed"] - ex["bs_fg"]).mean()),
                   "max_abs_diff": float((ex["avg_bat_speed"] - ex["bs_fg"]).abs().max())}

# ------------------------------------------------------------------ who drives Paper 1's lagged signal: eligibility strata
strata = {}
for name, m in (("F1-eligible (regulars)", L["f1_elig"]), ("not F1-eligible", ~L["f1_elig"])):
    d = L[m]
    strata[name] = fit(d, "nx_ob")
    strata[name]["oblique_rate_per_100"] = float(100 * d["p1_any_ob"].mean())
    strata[name]["bat_speed_mean"] = float(d["avg_bat_speed"].mean())
res["paper1_lagged_by_eligibility"] = strata
d = L.copy()
d["elig"] = d["f1_elig"].astype(int)
r = smf.gee("nx_ob ~ avg_bat_speed * elig + " + ADJ_P1, groups="mlbam", data=d.sort_values(["mlbam", "season"]), family=Poisson(),
            cov_struct=Exchangeable(), offset=d.sort_values(["mlbam", "season"])["nx_logoff"]).fit()
res["paper1_lagged_eligibility_interaction_p"] = float(r.pvalues["avg_bat_speed:elig"])
# by index season
res["paper1_lagged_by_index_season"] = {int(y): fit(L[L["ns"] == y], "nx_ob", adj=ADJ_P1.replace(" + C(season_f)", "")) for y in (2024, 2025, 2026)}
res["f1_eligible_by_index_season_p1labels"] = {int(y): fit(E[E["ns"] == y], "p1_any_ob", adj=ADJ_F1.replace(" + C(season_f)", ""), kind="logit_cluster") for y in (2025, 2026)}

json.dump(res, open(OUT, "w"), indent=1, default=str)
for blk in ("steps", "reverse"):
    print("==", blk)
    for k, v in res[blk].items():
        print(f"  {v['per_SD']:.2f} ({v['lo']:.2f}-{v['hi']:.2f}) P={v['p']:.3f} n={v['n']} ev={v['events']}  {k}")
print("== labels", json.dumps(res["labels"], indent=1))
print("== exposure", res["exposure"])
print("== Paper 1 lagged by eligibility", {k: (round(v['per_SD'], 2), round(v['lo'], 2), round(v['hi'], 2), v['n'], v['events'], round(v['oblique_rate_per_100'], 1), round(v['bat_speed_mean'], 2)) for k, v in strata.items()},
      "interaction P", round(res["paper1_lagged_eligibility_interaction_p"], 3))
print("== by index season", {k: (round(v['per_SD'], 2), round(v['lo'], 2), round(v['hi'], 2), v['events']) for k, v in res["paper1_lagged_by_index_season"].items()})
print("== F1-eligible by season, P1 labels", {k: (round(v['per_SD'], 2), round(v['lo'], 2), round(v['hi'], 2), v['events']) for k, v in res["f1_eligible_by_index_season_p1labels"].items()})
print("discordant", len(rows))
