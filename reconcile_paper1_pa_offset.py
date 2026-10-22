"""Post hoc: does playing time (exposure) explain part of Paper 1's lagged association? Run from the Hot_Hand_Bat_Tracking folder."""
import sys, os, runpy, io, contextlib, json
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from statsmodels.genmod.families import Poisson
from statsmodels.genmod.cov_struct import Exchangeable
sys.argv=["x","/home/claude/p1","repo/data_public/il_transactions_hitters_2024_2026.txt","repo/data_public/il_transactions_hitters_2024_2026_corrected.txt","inputs_fangraphs_not_redistributed/Fangraphs Bat Tracking ALL.csv","inputs_fangraphs_not_redistributed/Fangraphs Position Player Raw 2000-2025.xlsx","inputs_fangraphs_not_redistributed/fangraphs_batting_2026.csv","/tmp/rp1.json"]
with contextlib.redirect_stdout(io.StringIO()):
    g=runpy.run_path("repo/reconcile_paper1.py")
L,ADJ,SD=g["L"],g["ADJ_P1"],g["SD"]
def pfit(d,off,lab,y="nx_ob"):
    d=d.sort_values(["mlbam","season"])
    r=smf.gee(f"{y} ~ avg_bat_speed + {ADJ}",groups="mlbam",data=d,family=Poisson(),cov_struct=Exchangeable(),offset=off.loc[d.index]).fit()
    b,se=r.params["avg_bat_speed"]*SD,r.bse["avg_bat_speed"]*SD
    print(f"{lab:55s} {np.exp(b):.2f} ({np.exp(b-1.96*se):.2f}-{np.exp(b+1.96*se):.2f}) n={len(d)} ev={int((d[y]>0).sum())}")
    return dict(per_SD=float(np.exp(b)),lo=float(np.exp(b-1.96*se)),hi=float(np.exp(b+1.96*se)),n=int(len(d)))
d=L[L["pa_Y"].fillna(0)>0].copy()
d["off_pa"]=np.log(d["pa_Y"])
out={}
out["days_offset_PA_Y_gt0"]=pfit(d,d["nx_logoff"],"season-days offset, hitters with PA in T+1")
out["PA_Y_offset"]=pfit(d,d["off_pa"],"log(PA in T+1) offset (descriptive: PA is reduced by injury)")
d25=d[d["ns"]>=2025]
out["days_offset_2025_26"]=pfit(d25,d25["nx_logoff"],"season-days offset, 2025-26")
out["PA_Y_offset_2025_26"]=pfit(d25,d25["off_pa"],"log(PA in T+1) offset, 2025-26")
print("corr bat speed T with PA T+1:", round(d[["avg_bat_speed","pa_Y"]].corr().iloc[0,1],3), " among not covered:", round(d[~d["inU"]][["avg_bat_speed","pa_Y"]].corr().iloc[0,1],3))
out["corr_bs_paY"]=float(d[["avg_bat_speed","pa_Y"]].corr().iloc[0,1])
json.dump(out,open("repo/results/reconcile_paper1_pa_offset.json","w"),indent=1)
# expected exposure not affected by the outcome-season injury: prior-season PA as offset, and log(prior PA) as covariate
e=L[L["pa_prev"].fillna(0)>0].copy(); e["off_prev"]=np.log(e["pa_prev"]); e["lpa_prev"]=np.log(e["pa_prev"])
out["prior_PA_offset_all"]=pfit(e,e["off_prev"],"log(prior-season PA) offset, all")
e25=e[e["ns"]>=2025]; out["prior_PA_offset_2025_26"]=pfit(e25,e25["off_prev"],"log(prior-season PA) offset, 2025-26")
def cfit(d,lab):
    d=d.sort_values(["mlbam","season"])
    r=smf.gee(f"nx_ob ~ avg_bat_speed + lpa_prev + {ADJ}",groups="mlbam",data=d,family=Poisson(),cov_struct=Exchangeable(),offset=d["nx_logoff"]).fit()
    b,se=r.params["avg_bat_speed"]*SD,r.bse["avg_bat_speed"]*SD
    print(f"{lab:55s} {np.exp(b):.2f} ({np.exp(b-1.96*se):.2f}-{np.exp(b+1.96*se):.2f}) n={len(d)}  coef log prior PA {r.params['lpa_prev']:.2f} ({r.params['lpa_prev']-1.96*r.bse['lpa_prev']:.2f} to {r.params['lpa_prev']+1.96*r.bse['lpa_prev']:.2f})")
    return dict(per_SD=float(np.exp(b)),lo=float(np.exp(b-1.96*se)),hi=float(np.exp(b+1.96*se)),coef_log_prior_pa=float(r.params['lpa_prev']))
out["log_prior_PA_covariate_all"]=cfit(e,"log(prior-season PA) as covariate, all")
out["log_prior_PA_covariate_2025_26"]=cfit(e25,"log(prior-season PA) as covariate, 2025-26")
json.dump(out,open("repo/results/reconcile_paper1_pa_offset.json","w"),indent=1)
