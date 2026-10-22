"""Post hoc additions to F1 (H1, prior-season bat speed and oblique placement) after the reconciliation with Paper 1.

Not prespecified. Run from the Hot_Hand_Bat_Tracking folder after reconcile_paper1.py:
    python3 repo/ajsm_f1_post_hoc.py   (Paper 1 folder: P1_DIR, default ../Bat-Tracking + Injuries beside this folder)
Uses the 503 F1 hitter-seasons that are also in Paper 1's lagged panel (1 F1 hitter-season is not), the F1 design
(logistic, hitter-clustered; age, prior-season PA, prior IL, season) and the prespecified IL file (record-date dating).
Writes repo/results/ajsm_f1_post_hoc.json, read by paper/ajsm/build_ajsm.py.
  position     F1 plus position group (from Paper 1's MLB Stats API position data; the prespecified catcher
               covariate could not be built from the FanGraphs file)
  no_contusion F1 without the one oblique-family placement that is a contusion (Brenton Doyle, 2026)
  by_season    F1 in index season 2025 and 2026 separately
  power        power of F1 to detect the association Paper 1's design gives in the same index seasons (1.35 per 2.68 mph)
"""
import contextlib
import io
import json
import os
import runpy
import sys

import numpy as np
from scipy.stats import norm

P1 = os.environ.get("P1_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Bat-Tracking + Injuries")
sys.argv = ["x", P1, "repo/data_public/il_transactions_hitters_2024_2026.txt",
            "repo/data_public/il_transactions_hitters_2024_2026_corrected.txt",
            "inputs_fangraphs_not_redistributed/Fangraphs Bat Tracking ALL.csv",
            "inputs_fangraphs_not_redistributed/Fangraphs Position Player Raw 2000-2025.xlsx",
            "inputs_fangraphs_not_redistributed/fangraphs_batting_2026.csv", "/tmp/reconcile_paper1_tmp.json"]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reconcile_paper1.py"))
L, fit, ADJ_F1 = g["L"], g["fit"], g["ADJ_F1"]
E = L[(L["ns"] >= 2025) & L["f1_elig"]].copy()


def per_mph(r):
    return {"OR": r["per_mph"], "lo": r["lo"] ** (1 / 2.68), "hi": r["hi"] ** (1 / 2.68), "p": r["p"], "n": r["n"], "events": r["events"]}


out = {"note": __doc__.split("\n\n")[0], "n_overlap": int(len(E))}
out["base"] = per_mph(fit(E, "y_obl_orig", x="bs_fg", adj=ADJ_F1, kind="logit_cluster", scale=2.68))
out["position"] = per_mph(fit(E, "y_obl_orig", x="bs_fg", adj=ADJ_F1 + " + C(pos_group)", kind="logit_cluster", scale=2.68))
E["y_nc"] = E["y_obl_orig"]
E.loc[(E["mlbam"] == 686668) & (E["ns"] == 2026), "y_nc"] = 0
out["no_contusion"] = per_mph(fit(E, "y_nc", x="bs_fg", adj=ADJ_F1, kind="logit_cluster", scale=2.68))
out["by_season"] = {str(y): per_mph(fit(E[E["ns"] == y], "y_obl_orig", x="bs_fg", adj=ADJ_F1.replace(" + C(season_f)", ""),
                                        kind="logit_cluster", scale=2.68)) for y in (2025, 2026)}
se_mph = np.log(out["base"]["hi"] / out["base"]["lo"]) / (2 * 1.96)
target = 1.35 ** (1 / 2.68)
out["power"] = {"target_per_mph": target, "target_per_2.68mph": 1.35, "se_per_mph": float(se_mph),
                "power": float(norm.cdf(np.log(target) / se_mph - 1.96) + norm.cdf(-np.log(target) / se_mph - 1.96))}
json.dump(out, open("repo/results/ajsm_f1_post_hoc.json", "w"), indent=1)
for k in ("base", "position", "no_contusion"):
    r = out[k]
    print(k, f"{r['OR']:.3f} ({r['lo']:.3f}-{r['hi']:.3f}) n={r['n']} ev={r['events']}")
for y, r in out["by_season"].items():
    print(y, f"{r['OR']:.3f} ({r['lo']:.3f}-{r['hi']:.3f}) n={r['n']} ev={r['events']}")
print("power", out["power"])
