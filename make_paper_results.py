"""Merge every estimate the paper and abstract quote into one paper_results.json.

python3 make_paper_results.py DATA_DIR OUT3_DIR PRIMARY_JSON OUT_JSON
  OUT3_DIR      analysis3.py output (a1.json ... a10.json, null_mc.json)
  PRIMARY_JSON  analysis2.py output (results_adjusted.json / results2.json)
The 'abstract' block holds each number exactly as printed in the abstract; render_abstract.py fills the template
from it and verify_paper.py checks it against an independent recomputation.
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np

import hot_streak_study as H

data_dir, o3, prim_path, out_path = sys.argv[1:5]
J = lambda n: json.load(open(os.path.join(o3, n)))
P = json.load(open(prim_path))
p50 = P["rec50_fwd50"]
forms = p50["woba_forms"] if "woba_forms" in p50 else p50["woba"]["forms"]
pers_primary = {r["metric"]: r for r in p50["persistence"]}
mc = J("null_mc.json")
A = {f"A{k}": J(f"a{k}.json") for k in range(1, 11)}

d = H.load(data_dir)
n_pitches, n_hitters = int(len(d)), int(d.loc[d["woba_denom"] == 1, "batter"].nunique())
h = hashlib.sha256()
for f in sorted(glob.glob(os.path.join(data_dir, "*.parquet"))):
    h.update(open(f, "rb").read())

pct = lambda x: f"{100 * x:.1f}%"
pts = lambda x: f"{x:+.1f}".replace("-", "−")
num = lambda x, k=1: f"{x:.{k}f}".replace("-", "−")

# persistence: actual slope minus the mean of 20 shuffled-null slopes; SE from the analytic formula of the primary run
PERS = {m: {"persistence": mc[m]["persistence_mean_null"], "se": pers_primary[m]["se"],
            "slope_actual": mc[m]["slope_actual"], "null_mean_20": mc[m]["null_mean"],
            "null_sd_across_shuffles": mc[m]["null_sd_across_shuffles"], "single_draw_persistence": pers_primary[m]["persistence"]}
        for m in mc}
a1, a2, a3, a5, a7, a8 = A["A1"], A["A2"], A["A3"], A["A5"], A["A7"], A["A8"]
fixed = a1["horizons_fixed_sample"]
sw, rs = forms["swing_only"], forms["results_only"]
surv = [a2["ipw"]["bottom"], a2["cross_season"]["bottom"], a2["regulars_400pa"]["bottom"]]
pers5 = a5["persistence"]
other_p = max(v["persistence"] for m, v in pers5.items() if not m.startswith("ff_"))
deliv = [pers5[m]["persistence"] for m in ("ff_spin", "ff_ext", "ff_relz", "ff_arm") if m in pers5]
st = a5["starters_results_vs_velocity"]
il = a8["risk"]["bat_speed_only"]["z_bs"]
A7p, A7r, OB = a7["prespecified"], a7["refined_exploratory"], "option_b_vs_strict_platoon_runs_per50"
a10 = A["A10"]
# bottom decile of the swing-form index across every specification (primary, A2 survivorship, A3 baselines, A10 windows/filters)
spec_bot = [sw["deciles"]["bump"][0], a2["ipw"]["bottom"], a2["cross_season"]["bottom"], a2["regulars_400pa"]["bottom"]]
A3B = lambda s: [b for b in a3[s] if isinstance(a3[s][b], dict) and "swing_only" in a3[s][b]]
spec_bot += [a3[s][b]["swing_only"]["bottom"] for s in a3 for b in A3B(s)]
spec_bot += [a10[k]["swing_form"]["bottom"] for k in ("w100_100", "days14_14", "days28_28", "competitive_filter_off", "park_week_adjustment_off")]
res_top = [rs["deciles"]["bump"][9]] + [a3[s][b]["results_only"]["top"] for s in a3 for b in A3B(s)]
res_top += [a10[k]["results_form"]["top"] for k in ("w100_100", "days14_14", "days28_28", "competitive_filter_off", "park_week_adjustment_off")]
spreads = [a10[f"season_{y}"]["swing_form"]["top"] - a10[f"season_{y}"]["swing_form"]["bottom"] for y in (2024, 2025, 2026)]

abstract = {
    "n_pitches": f"{n_pitches:,}", "n_hitters": f"{n_hitters:,}",
    "p_woba": pct(PERS["woba"]["persistence"]), "se_woba": num(100 * PERS["woba"]["se"]),
    "p_xwoba": pct(PERS["xwoba"]["persistence"]),
    "p_contact_max": pct(max(PERS[m]["persistence"] for m in ["ev", "hard_hit", "barrel"])),
    "p_k": pct(max(PERS[m]["persistence"] for m in ["ev", "hard_hit", "barrel", "k_rate"])),
    "p_bat_speed": pct(PERS["bat_speed"]["persistence"]), "se_bat_speed": num(100 * PERS["bat_speed"]["se"]),
    "p_swing_len": pct(PERS["swing_len"]["persistence"]), "p_fast_swing": pct(PERS["fast_swing"]["persistence"]),
    "p_z_swing": pct(PERS["z_swing"]["persistence"]), "p_chase": pct(PERS["chase"]["persistence"]),
    "decay_25": pct(fixed["25"]["bat_speed"]["persistence"]), "decay_200": pct(fixed["200"]["bat_speed"]["persistence"]),
    "cp_absorbed": f"{100 * (1 - a1['changepoint']['share_surviving']):.0f}%",
    "top_swing": num(sw["deciles"]["bump"][9]), "top_swing_se": num(sw["deciles"]["se"][9]),
    "bot_swing": num(-sw["deciles"]["bump"][0]), "bot_swing_se": num(sw["deciles"]["se"][0]),
    "top_results": num(rs["deciles"]["bump"][9]), "top_results_se": num(rs["deciles"]["se"][9]),
    "surv_lo": num(-max(surv)), "surv_hi": num(-min(surv)),
    "marcel_top": num(a3["primary_sample"]["marcel"]["swing_only"]["top"]),
    "marcel_bot": num(-a3["primary_sample"]["marcel"]["swing_only"]["bottom"]),
    "p_velo": pct(pers5["ff_velo"]["persistence"]), "p_pitch_other_max": pct(other_p),
    "p_deliv_lo": f"{100 * min(deliv):.0f}", "p_deliv_hi": f"{100 * max(deliv + [pers5['ff_velo']['persistence']]):.0f}%",
    "fg_bot": num(-a3["primary_sample"]["fangraphs_marcel"]["swing_only"]["bottom"]),
    "velo_starts": num(-st["velo_effect"]), "velo_starts_se": num(st["velo_se"]),
    "results_starts": num(st["results_effect"]), "results_starts_se": num(st["results_se"]),
    "lhb_pen": num(P["platoon"]["LHB_league_split_pts"]), "rhb_pen": num(P["platoon"]["RHB_league_split_pts"]),
    "form_flips_lo": f"{100 * min(A7p['share_form_flips_call'], A7r['share_form_flips_call']):.0f}",
    "form_flips_hi": f"{100 * max(A7p['share_form_flips_call'], A7r['share_form_flips_call']):.0f}%",
    "hot_changes_lo": f"{100 * min(A7p[OB]['share_differs_from_platoon'], A7r[OB]['share_differs_from_platoon']):.0f}",
    "hot_changes_hi": f"{100 * max(A7p[OB]['share_differs_from_platoon'], A7r[OB]['share_differs_from_platoon']):.0f}%",
    "hot_cost_lo": num(-max(A7p[OB]['mean'], A7r[OB]['mean']), 2), "hot_cost_hi": num(-min(A7p[OB]['mean'], A7r[OB]['mean']), 2),
    "spec_bot_lo": num(-max(spec_bot)), "spec_bot_hi": num(-min(spec_bot)),
    "season_spread_lo": f"{min(spreads):.0f}", "season_spread_hi": f"{max(spreads):.0f}",
    "res_bot": num(rs["deciles"]["bump"][0]), "res_bot_se": num(rs["deciles"]["se"][0]),
    "res_top_lo": num(min(res_top)), "res_top_hi": num(max(res_top)),
    "velo_sd_mph": num(st["actual"]["sd_velo_dev_mph"]),
    "il_or": num(il["OR_per_minus1SD"], 2), "il_lo": num(il["ci"][0], 2), "il_hi": num(il["ci"][1], 2),
}
TABLE = ["swing_len", "bat_speed", "fast_swing", "z_swing", "chase", "whiff", "barrel", "k_rate", "ev", "xwoba", "hard_hit", "woba"]
for m in TABLE:
    abstract[f"t_{m}"] = f"{num(100 * PERS[m]['persistence'])}% ({num(100 * PERS[m]['se'])})"

SIM = {}
for tag in ("hot", "null"):
    p = os.path.join(os.path.dirname(o3), f"out_sim_{tag}", "results.json")
    if os.path.exists(p):
        r = json.load(open(p))["results"]["rec50_fwd50"]["persistence"]
        SIM[tag] = {x["metric"]: {"persistence": x["persistence"], "se": x["se"]} for x in r if x["metric"] in ("bat_speed", "woba", "chase")}
abstract["sim_hot_bs"] = pct(SIM["hot"]["bat_speed"]["persistence"]) if SIM else "n/a"
abstract["sim_null_bs"] = pct(SIM["null"]["bat_speed"]["persistence"]) if SIM else "n/a"
# A11 (offseason analysis, plan addendum C): offseason carry-over and held-out step follow-up; optional until a11.json exists
A11 = None
p11 = os.path.join(o3, "a11.json")
if os.path.exists(p11):
    a11 = json.load(open(p11))
    rb, rn, dec = a11["real"]["A11b"], a11["real"]["A11a"]["next"]["pooled"], a11["decision"]
    c11 = a11["real"]["A11c"]
    A11 = {"b": rb["b"], "b_se": rb["se"], "b_n": rb["n"], "b_null": rb["b_null"], "b_excess": rb["excess"],
           "b_excess_se": rb["excess_se"], "R_next": rn["R"], "R_next_se": rn["se"], "R_next_n": rn["n"],
           "half_life_pa": c11["ar_rw"]["half_life_pa"], "half_life_ci": [c11["boot"]["half_life_pa"]["lo"], c11["boot"]["half_life_pa"]["hi"]],
           "sigma2_rw": c11["ar_rw"]["sigma2_rw"], "sigma2_rw_ci": [c11["boot"]["sigma2_rw"]["lo"], c11["boot"]["sigma2_rw"]["hi"]],
           "branch": dec["branch"], "branch_before_robustness": dec["primary"]["branch"], "robustness": dec["robustness"],
           "plan_sha16": a11["meta"]["plan_sha16"], "signature": a11["signature"]}
    abstract["xs_carry"] = pct(rb["excess"])
    abstract["xs_carry_null"] = pct(rb["b_null"])
    abstract["half_life"] = f"{c11['ar_rw']['half_life_pa']:.0f}"

# PA horizons in calendar weeks (descriptive, horizon_weeks.py; added 2026-10-01)
pw = os.path.join(o3, "horizon_weeks.json")
HW = json.load(open(pw)) if os.path.exists(pw) else None
if HW is not None:
    abstract["weeks_200"] = f"{HW['200']['median_days'] / 7:.0f}"

out = {"meta": {"built": "2026-09-30", "statcast_sha16": h.hexdigest()[:16],
                "statcast_files": [os.path.basename(f) for f in sorted(glob.glob(os.path.join(data_dir, "*.parquet")))], "plan": "PLAN_2026-09-30.md",
                "n_pitches": n_pitches, "n_hitters": n_hitters,
                "persistence_convention": "actual slope minus the mean of 20 within-hitter-season shuffles; SE = sqrt(SE_actual^2 + SE_null^2) from the primary run",
                "exploratory": ["A1 non-overlapping block autocorrelation (definition differs from the pre-registration-free check quoted in the 2026-09-30 abstract draft)"]},
       "primary": {"persistence": PERS, "forms": forms, "platoon": P["platoon"]},
       **A, **({"A11": A11} if A11 is not None else {}), **({"horizon_weeks": HW} if HW is not None else {}), "simulation": SIM, "abstract": abstract}
json.dump(out, open(out_path, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(abstract, indent=1, ensure_ascii=False))
