"""Post hoc companion to reconcile_paper1.py (run from the Hot_Hand_Bat_Tracking folder; P1_DIR = folder with Paper 1 data). Logistic models are cluster-robust with a binary outcome."""
import os
import sys, runpy, itertools, json
sys.argv=["x", os.environ.get("P1_DIR", "/home/claude/p1"), "repo/data_public/il_transactions_hitters_2024_2026.txt", "repo/data_public/il_transactions_hitters_2024_2026_corrected.txt", "inputs_fangraphs_not_redistributed/Fangraphs Bat Tracking ALL.csv", "inputs_fangraphs_not_redistributed/Fangraphs Position Player Raw 2000-2025.xlsx", "inputs_fangraphs_not_redistributed/fangraphs_batting_2026.csv", "/tmp/reconcile_paper1_tmp.json"]
import io, contextlib
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    g=runpy.run_path("repo/reconcile_paper1.py")
L,fit,ADJ_P1,ADJ_F1=g["L"],g["fit"],g["ADJ_P1"],g["ADJ_F1"]
U=L[L["inU"] & L["age_f1"].notna() & L["pa100"].notna()].copy()
U["y_p1"]=U["p1_any_ob"]; U["y_ours"]=U["y_obl_corr"]
rows=[]
for seasons,sample,cov,lab in itertools.product(["2024-26","2025-26"],["covered, >=100 comp","F1-eligible"],["P1","F1"],["P1","ours"]):
    d=U if seasons=="2024-26" else U[U["ns"]>=2025]
    if sample=="F1-eligible": d=d[d["f1_elig"]]
    adj=ADJ_P1 if cov=="P1" else ADJ_F1
    if cov=="F1" and seasons=="2024-26": adj=adj  # il_prev is 0 for T=2023 (IL file starts 2024); season factor absorbs it
    r=fit(d,"y_p1" if lab=="P1" else "y_ours",adj=adj,kind="logit_cluster")
    rows.append(dict(seasons=seasons,sample=sample,cov=cov,labels=lab,**{k:r[k] for k in ("per_SD","lo","hi","p","n","events")}))
import pandas as pd
T=pd.DataFrame(rows); print(T.round(3).to_string())
# covariate-by-covariate on F1-eligible 2025-26, P1 labels
E=U[(U["ns"]>=2025)&U["f1_elig"]]
base="age + bmi + C(pos_group) + switch + lefty + C(season_f)"
var={"P1 set":base,"- bmi":base.replace("bmi + ",""),"- position":base.replace("C(pos_group) + ",""),"- handedness":base.replace("switch + lefty + ",""),
     "+ prior PA":base+" + pa100","+ prior IL":base+" + il_prev","+ prior PA + prior IL":base+" + pa100 + il_prev",
     "F1 set":ADJ_F1,"F1 + bmi":ADJ_F1+" + bmi","F1 + position":ADJ_F1+" + C(pos_group)","F1 + handedness":ADJ_F1+" + switch + lefty","F1 - prior PA":ADJ_F1.replace("pa100 + ",""),"F1 - prior IL":ADJ_F1.replace("il_prev + ",""),"none (season only)":"C(season_f)"}
cv=[]
for k,a in var.items():
    for lab in ("y_p1","y_ours"):
        r=fit(E,lab,adj=a,kind="logit_cluster"); cv.append(dict(covariates=k,labels=lab,**{x:r[x] for x in ("per_SD","lo","hi","p")}))
C=pd.DataFrame(cv); print(C.round(3).to_string())
# same covariate scan on the full Paper 1 lagged panel (count, offset)
cv2=[]
for k,a in var.items():
    if "pa100" in a or "il_prev" in a or "age_f1" in a:
        d=L[L["pa100"].notna() & L["age_f1"].notna()]
    else:
        d=L
    r=fit(d,"nx_ob",adj=a.replace("age_f1","age"),kind="poisson_offset"); cv2.append(dict(covariates=k,**{x:r[x] for x in ("per_SD","lo","hi","p","n")}))
C2=pd.DataFrame(cv2); print(C2.round(3).to_string())
json.dump({"factorial":rows,"covariates_F1elig":cv,"covariates_P1_full":cv2},open("repo/results/reconcile_paper1_factorial.json","w"),indent=1)
