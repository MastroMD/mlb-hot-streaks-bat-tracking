"""Post hoc companion to reconcile_paper1.py (run from the Hot_Hand_Bat_Tracking folder; P1_DIR = folder with Paper 1 data). Logistic models are cluster-robust with a binary outcome."""
import os
import sys, runpy, io, contextlib, json
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from statsmodels.genmod.families import Poisson
from statsmodels.genmod.cov_struct import Exchangeable
from scipy.stats import norm
sys.argv=["x", os.environ.get("P1_DIR", "/home/claude/p1"), "repo/data_public/il_transactions_hitters_2024_2026.txt", "repo/data_public/il_transactions_hitters_2024_2026_corrected.txt", "inputs_fangraphs_not_redistributed/Fangraphs Bat Tracking ALL.csv", "inputs_fangraphs_not_redistributed/Fangraphs Position Player Raw 2000-2025.xlsx", "inputs_fangraphs_not_redistributed/fangraphs_batting_2026.csv", "/tmp/reconcile_paper1_tmp.json"]
with contextlib.redirect_stdout(io.StringIO()):
    g=runpy.run_path("repo/reconcile_paper1.py")
L,fit,ADJ_P1,ADJ_F1,SD=g["L"],g["fit"],g["ADJ_P1"],g["ADJ_F1"],g["SD"]
out={}
# 1 heterogeneity across index seasons (lagged, Paper 1 design)
d=L.sort_values(["mlbam","season"]).copy()
d["bs_c"]=d["avg_bat_speed"]-d["avg_bat_speed"].mean()
r0=smf.gee("nx_ob ~ bs_c + "+ADJ_P1,groups="mlbam",data=d,family=Poisson(),cov_struct=Exchangeable(),offset=d["nx_logoff"]).fit()
r1=smf.gee("nx_ob ~ bs_c*C(season_f) + age + bmi + C(pos_group) + switch + lefty",groups="mlbam",data=d,family=Poisson(),cov_struct=Exchangeable(),offset=d["nx_logoff"]).fit()
terms=[t for t in r1.params.index if t.startswith("bs_c:")]
w=r1.wald_test(" , ".join(f"{t} = 0" for t in terms), scalar=True)
out["lag_heterogeneity_P"]=float(w.pvalue)
d["trend"]=d["season"]-2024
r2=smf.gee("nx_ob ~ bs_c*trend + age + bmi + C(pos_group) + switch + lefty + C(season_f)",groups="mlbam",data=d,family=Poisson(),cov_struct=Exchangeable(),offset=d["nx_logoff"]).fit()
out["lag_trend_per_season_ratio_per_SD"]=float(np.exp(r2.params["bs_c:trend"]*SD)); out["lag_trend_P"]=float(r2.pvalues["bs_c:trend"])
# 2 power of F1 (SE per 2.68 mph from F1 design on the overlap)
S=json.load(open("repo/results/reconcile_paper1.json"))
se=S["steps"]["7 + F1 model (logistic, cluster-robust)"]["se_per_SD"]
out["f1_se_per_SD"]=se
out["f1_power"]={str(t):float(norm.cdf(np.log(t)/se-1.96)+norm.cdf(-np.log(t)/se-1.96)) for t in (1.2,1.3,1.39,1.47)}
# 3 pooled estimates without the half-season 2023 exposure
L25=L[L["ns"]>=2025]
out["p1_2025_26_P1cov"]=fit(L25,"nx_ob")
out["p1_2025_26_plus_priorPA"]=fit(L25[L25["pa100"].notna()],"nx_ob",adj=ADJ_P1+" + pa100")
out["p1_2025_26_logit_P1cov"]=fit(L25,"p1_any_ob",kind="logit_cluster")
# 4 F1-eligible vs not, 2025-26
for nm,m in (("elig",L25["f1_elig"]),("not_elig",~L25["f1_elig"])):
    out[f"p1_2025_26_{nm}"]=fit(L25[m],"nx_ob"); out[f"p1_2025_26_{nm}"]["bs_sd"]=float(L25[m]["avg_bat_speed"].std())
# 5 contribution of the 2024 index season
out["p1_2024_only"]=fit(L[L["ns"]==2024],"nx_ob",adj=ADJ_P1.replace(" + C(season_f)",""))
json.dump(out,open("repo/results/reconcile_paper1_extra.json","w"),indent=1)
for k,v in out.items():
    if isinstance(v,dict) and "per_SD" in v: print(f"{k:28s} {v['per_SD']:.2f} ({v['lo']:.2f}-{v['hi']:.2f}) P={v['p']:.3f} n={v['n']} ev={v['events']}" + (f" sd={v['bs_sd']:.2f}" if 'bs_sd' in v else ""))
    else: print(k, v)
# ---- added after the independent check
add={}
Lc=L[L["inU"]]
Lc25=Lc[Lc["ns"]>=2025]
add["covered_2025_26_poisson_P1labels"]=fit(Lc25,"nx_ob")
add["covered_2025_26_poisson_AJSMlabels"]=fit(Lc25.assign(y=Lc25["y_obl_corr"]),"y")
add["covered_all_poisson_AJSMlabels"]=fit(Lc.assign(y=Lc["y_obl_corr"]),"y")
add["covered_2024_poisson_AJSMlabels"]=fit(Lc[Lc["ns"]==2024].assign(y=Lc[Lc["ns"]==2024]["y_obl_corr"]),"y",adj=ADJ_P1.replace(" + C(season_f)",""))
E=L[(L["ns"]>=2025)&L["f1_elig"]]
add["F1_overlap_P1_poisson"]=fit(E,"nx_ob")
for y in (2025,2026):
    add[f"F1_design_AJSMlabels_{y}"]=fit(E[E["ns"]==y],"y_obl_orig",x="bs_fg",adj=ADJ_F1.replace(" + C(season_f)",""),kind="logit_cluster")
add["power_vs_1.35"]=float(norm.cdf(np.log(1.35)/se-1.96)+norm.cdf(-np.log(1.35)/se-1.96))
# same-season trend (Paper 1 panel)
P=pd.read_csv(os.path.join(os.environ.get("P1_DIR","/home/claude/p1"),"data/audit_2026-10-06/panel.csv")).sort_values(["mlbam","season"])
P["logoff"]=np.log(P.season_days); P["switch"]=(P.bat_side=="S").astype(int); P["lefty"]=(P.bat_side=="L").astype(int)
P["season_f"]=P.season.astype(str); P["pos_group"]=pd.Categorical(P.pos_group,categories=["C","CIF","MIF","OF","DH"])
P["bs_c"]=P.avg_bat_speed-P.avg_bat_speed.mean(); P["trend"]=P.season-2023
r=smf.gee("dx_oblique_strain ~ bs_c*trend + age + bmi + C(pos_group) + switch + lefty + C(season_f)",groups="mlbam",data=P,family=Poisson(),cov_struct=Exchangeable(),offset=P["logoff"]).fit()
add["same_season_trend_ratio_per_SD"]=float(np.exp(r.params["bs_c:trend"]*SD)); add["same_season_trend_P"]=float(r.pvalues["bs_c:trend"])
# lagged without core-muscle placements (athletic pubalgia)
ST=g["ST"]; core=ST[(ST["site_fine"]=="oblique_strict")&ST["diagnosis"].str.contains("core",case=False,na=False)][["mlbam","season"]].drop_duplicates()
cm=set(zip(core.mlbam,core.season))
Lnc=L.copy(); Lnc["y"]=[0 if (m,s) in cm else v for m,s,v in zip(Lnc.mlbam,Lnc.ns,Lnc.nx_ob)]
add["lagged_no_core_all"]=fit(Lnc,"y"); add["lagged_no_core_2025_26"]=fit(Lnc[Lnc["ns"]>=2025],"y")
add["core_muscle_player_seasons"]=sorted([list(map(int,x)) for x in cm])
# timing of 2026 lagged events relative to the July 25 draft
ob26=ST[(ST["site_fine"]=="oblique_strict")&(ST["season"]==2026)]
ids26=set(L.loc[(L["ns"]==2026)&(L["nx_ob"]>0),"mlbam"])
o=ob26[ob26["mlbam"].isin(ids26)]
add["lag2026_events_before_jul25"]=int((o["start"]<pd.Timestamp("2026-07-25")).sum()); add["lag2026_events_total"]=int(len(o))
# oblique rate by index season (lagged panel)
add["lag_rate_per100_by_ns"]={int(k):float(v) for k,v in (100*L.groupby("ns")["p1_any_ob"].mean()).items()}
out["added_after_check"]=add
json.dump(out,open("repo/results/reconcile_paper1_extra.json","w"),indent=1)
for k,v in add.items():
    if isinstance(v,dict) and "per_SD" in v: print(f"{k:40s} {v['per_SD']:.2f} ({v['lo']:.2f}-{v['hi']:.2f}) P={v['p']:.3f} n={v['n']} ev={v['events']}")
    else: print(k, v)
