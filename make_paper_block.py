"""Write the numbers the full paper prints into paper_results.json['paper'] (plan addendum D, PLAN_ADDENDUM_2026-10-02d.md).

python3 make_paper_block.py RESULTS_DIR [COMPANION_SYNTHESIS_JSON]

RESULTS_DIR holds paper_results.json and the files it was merged from (a11.json, b3.json, a7_pairs.csv,
results_adjusted.json, paper_counts.json). Every number the paper's text and tables print is formatted here, once, and
stored as a string with the key it came from. build_paper.py fills its template from this block and never computes a
number. The script leaves every other block of paper_results.json unchanged and stops if the abstract block's hash has
moved. COMPANION_SYNTHESIS_JSON is the companion paper's results file; without it the values stored by an earlier run
are kept (they are a fixed snapshot, hash recorded).
"""
import csv
import hashlib
import json
import os
import sys

R = sys.argv[1]
PR_PATH = os.path.join(R, "paper_results.json")
P = json.load(open(PR_PATH))
J = lambda n: json.load(open(os.path.join(R, n)))
sha16 = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
ABSTRACT_SHA16 = "93fc915627b12146"   # the submitted abstract's numbers (plan addendum D)
ah = hashlib.sha256(json.dumps(P["abstract"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
if ah != ABSTRACT_SHA16:
    sys.exit(f"abstract block changed ({ah} != {ABSTRACT_SHA16}); stop")

MINUS = "−"
def fix(s):
    """unicode minus, and no negative zero ("-0.00" prints as "0.00")"""
    if s.startswith("-") and not any(ch in "123456789" for ch in s):
        s = s[1:]
    return s.replace("-", MINUS)
pct = lambda x, k=1: fix(f"{100 * x:.{k}f}%")
pc = lambda x, k=1: fix(f"{100 * x:.{k}f}")          # percent value without the sign
num = lambda x, k=1: fix(f"{x:.{k}f}")
sgn = lambda x, k=1: fix(f"{x:+.{k}f}")
cnt = lambda x: f"{int(round(x)):,}"
se_ = lambda x, k=1: f"({num(x, k)})"

T = {}       # text values
SRC = {}     # key -> where it came from


def put(key, value, source):
    if key in T and T[key] != value:
        sys.exit(f"duplicate key with a different value: {key}")
    T[key] = value
    SRC[key] = source


A = P["abstract"]
for k, v in A.items():                               # the abstract's own strings, reused verbatim
    put(k, v, f"abstract.{k}")
put("extreme", f"{round((float(A['top_swing']) + float(A['bot_swing'])) / 2):.0f}", "abstract top_swing, bot_swing (mean, as render_abstract.py)")

# ------------------------------------------------------------------ data and design
cn = J("paper_counts.json")
put("n_pa", cnt(cn["n_pa"]), "paper_counts.n_pa")
put("n_games", cnt(cn["n_games"]), "paper_counts.n_games")
put("n_pitchers_all", cnt(cn["n_pitchers"]), "paper_counts.n_pitchers")
put("n_tracked_swings", cnt(cn["n_tracked_swings"]), "paper_counts.n_tracked_swings")
put("n_comp_swings", cnt(cn["n_competitive_swings"]), "paper_counts.n_competitive_swings")
put("n_batter_games", cnt(cn["n_batter_games"]), "paper_counts.n_batter_games")
put("n_hs300", cnt(cn["n_hitter_seasons_300pa"]), "paper_counts.n_hitter_seasons_300pa")
put("n_hitters300", cnt(cn["n_hitters_300pa"]), "paper_counts.n_hitters_300pa")
ps = cn["persistence_sample_bat_speed"]
put("n_pers_points", cnt(ps["n_points"]), "paper_counts.persistence_sample_bat_speed.n_points")
put("first_tracked_2024", cn["first_tracked_date"]["2024"], "paper_counts.first_tracked_date.2024")
put("n_games_2026", cnt(cn["n_games_by_season"]["2026"]), "paper_counts.n_games_by_season.2026")
# design constants (code: hot_streak_study.MIN_N, analysis3.A7_core; plan PLAN_2026-09-30.md)
DESIGN = {"min_comp": cn["min_n"]["comp"], "min_zone": cn["min_n"]["inzone"], "min_bip": cn["min_n"]["bip"],
          "min_season_pa": 300, "min_prior_pa": 200, "comp_pctile": 10, "comp_ev": 90, "savant_comp_floor": 60,
          "fast_mph": 75, "hard_ev": 95, "pair_min_pa": 150, "pair_min_starts": 20, "pair_costart_pct": 15,
          "pair_lhb_vs_rhp_pct": 70, "pair_rhb_vs_lhp_pct": 60, "k_split_lhb": 1000, "k_split_rhb": 2200,
          "p_same_starter_pct": 60, "p_same_pen_pct": 40, "woba_scale": 1.24, "n_shuffles": 20,
          "a1_shuffles": 10, "a10_shuffles": 5, "a5_shuffles": 10, "pitch_window": 200, "pitch_min_season": 1000,
          "starter_min_bf": 15, "il_days": 30, "il_return_days": 45, "gz_window": 25, "gz_tail_pct": 5,
          "gz_exclude": 50, "a11_window": 150, "a11_min_pa_y": 300, "a11_min_pa_next": 150, "a11_rnext_pa": 100,
          "a11_fit_share_pct": 60, "cp_min_games": 20, "a6_min_pa": 10, "a4_min_pairs": 15, "a2_regular_pa": 400,
          "a10_days_short": 14, "a10_days_long": 28, "a10_drop_days": 30,
          "sim_hitters": 160, "sim_games": 150, "cp_min_swings": 5, "pull_date": "2026-09-30",
          "sim_woba_pts": 12}
for k, v in DESIGN.items():
    put(f"d_{k}", f"{v:,}" if isinstance(v, int) else f"{v}", f"design.{k}")

# ------------------------------------------------------------------ 4.1 persistence (Table 1)
PERS = P["primary"]["persistence"]
put("null_bs", num(PERS["bat_speed"]["null_mean_20"], 2), "primary.persistence.bat_speed.null_mean_20")
put("null_woba", num(PERS["woba"]["null_mean_20"], 2), "primary.persistence.woba.null_mean_20")
put("act_bs", num(PERS["bat_speed"]["slope_actual"], 2), "primary.persistence.bat_speed.slope_actual")
put("act_woba", num(PERS["woba"]["slope_actual"], 2), "primary.persistence.woba.slope_actual")
put("woba_upper95", pct(PERS["woba"]["persistence"] + 1.96 * PERS["woba"]["se"]), "primary.persistence.woba (p + 1.96 se)")
put("null_mc_sd_bs", num(100 * PERS["bat_speed"]["null_sd_across_shuffles"], 1), "primary.persistence.bat_speed.null_sd_across_shuffles")
nulls = [v["null_mean_20"] for v in PERS.values()]
put("null_lo", num(min(nulls), 2), "primary.persistence.*.null_mean_20 (min)")
put("null_hi", num(max(nulls), 2), "primary.persistence.*.null_mean_20 (max)")

RA = J("results_adjusted.json")
LABEL = {"swing_len": "Swing length", "bat_speed": "Bat speed", "fast_swing": "Fast-swing rate", "z_swing": "In-zone swing rate",
         "chase": "Chase rate", "whiff": "Whiff rate", "barrel": "Barrel rate", "k_rate": "Strikeout rate", "ev": "Exit velocity",
         "xwoba": "xwOBA", "hard_hit": "Hard-hit rate", "woba": "wOBA"}
GROUP = {"bat_speed": "Swing mechanics", "swing_len": "Swing mechanics", "fast_swing": "Swing mechanics",
         "z_swing": "Swing decisions", "chase": "Swing decisions", "whiff": "Swing decisions",
         "barrel": "Contact quality", "ev": "Contact quality", "hard_hit": "Contact quality", "xwoba": "Results",
         "k_rate": "Results", "woba": "Results"}
ORDER = ["bat_speed", "swing_len", "fast_swing", "z_swing", "chase", "whiff", "ev", "hard_hit", "barrel", "k_rate", "xwoba", "woba"]
win = {w: {r["metric"]: r for r in RA[w]["persistence"]} for w in ("rec25_fwd25", "rec100_fwd50")}
T1 = {"header": ["Metric", "Group", "Actual slope", "Shuffled-null slope", "Persistence (SE)",
                 "25 PA → 25 PA", "100 PA → 50 PA"], "rows": []}
for m in ORDER:
    p = PERS[m]
    T1["rows"].append([LABEL[m], GROUP[m], num(p["slope_actual"], 3), num(p["null_mean_20"], 3),
                       f"{pct(p['persistence'])} {se_(100 * p['se'])}",
                       f"{pct(win['rec25_fwd25'][m]['persistence'])} {se_(100 * win['rec25_fwd25'][m]['se'])}",
                       f"{pct(win['rec100_fwd50'][m]['persistence'])} {se_(100 * win['rec100_fwd50'][m]['se'])}"])
w5050 = {r["metric"]: r for r in RA["rec50_fwd50"]["persistence"]}
LABEL4 = {"squared_up": ("Squared-up rate", "Contact quality"), "blast": ("Blast rate", "Contact quality"),
          "xwobacon": ("xwOBA on contact", "Contact quality"), "bb_rate": ("Walk rate", "Results")}
for m4, (lab4, grp4) in LABEL4.items():
    r = w5050[m4]
    T1["rows"].append([lab4 + " (single shuffle)", grp4, num(r["slope_actual"], 3), num(r["slope_null"], 3),
                       f"{pct(r['persistence'])} {se_(100 * r['se'])}",
                       f"{pct(win['rec25_fwd25'][m4]['persistence'])} {se_(100 * win['rec25_fwd25'][m4]['se'])}",
                       f"{pct(win['rec100_fwd50'][m4]['persistence'])} {se_(100 * win['rec100_fwd50'][m4]['se'])}"])
put("p_extra4_max", pct(max(w5050[m]["persistence"] for m in LABEL4)), "results_adjusted.rec50_fwd50 (squared_up, blast, xwobacon, bb_rate) max")
T1["n_points_note"] = cnt(ps["n_points"])
put("t1_n_points", cnt(ps["n_points"]), "paper_counts.persistence_sample_bat_speed.n_points")
put("t1_n_hitters", cnt(ps["n_hitters"]), "paper_counts.persistence_sample_bat_speed.n_hitters")
put("t1_n_hs", cnt(ps["n_hitter_seasons"]), "paper_counts.persistence_sample_bat_speed.n_hitter_seasons")

# A9 literature reconciliation (Table 4)
a9 = P["A9"]
ab = a9["albright_runs"]
ob, bsr = ab["on_base_by_PA"], ab["bat_speed_above_median_by_swing"]
put("alb_n_ob", cnt(ob["n_hitter_seasons"]), "A9.albright_runs.on_base_by_PA.n_hitter_seasons")
put("alb_z_ob", num(ob["mean_z"], 2), "A9.albright_runs.on_base_by_PA.mean_z")
put("alb_z_ob_sh", num(ob["mean_z_game_shuffle"], 2), "A9.albright_runs.on_base_by_PA.mean_z_game_shuffle")
put("alb_diff_ob", num(ob["mean_diff"], 2), "A9.albright_runs.on_base_by_PA.mean_diff")
put("alb_diff_ob_se", num(ob["se_diff"], 2), "A9.albright_runs.on_base_by_PA.se_diff")
put("alb_z_bs", num(bsr["mean_z"], 2), "A9.albright_runs.bat_speed_above_median_by_swing.mean_z")
put("alb_z_bs_sh", num(bsr["mean_z_game_shuffle"], 2), "A9.albright_runs.bat_speed_above_median_by_swing.mean_z_game_shuffle")
put("alb_diff_bs", num(bsr["mean_diff"], 2), "A9.albright_runs.bat_speed_above_median_by_swing.mean_diff")
put("alb_diff_bs_se", num(bsr["se_diff"], 2), "A9.albright_runs.bat_speed_above_median_by_swing.se_diff")
put("alb_n_bs", cnt(bsr["n_hitter_seasons"]), "A9.albright_runs.bat_speed_above_median_by_swing.n_hitter_seasons")
EV = J("e_review.json")
gz1 = a9["green_zwiebel"]["null_corrected"]                  # prespecified A9: one shuffle
gz = EV["E2_green_zwiebel_20_shuffles"]                       # exploratory refinement: 20 shuffles (plan addendum E)
put("gz1_bs_cold_woba", sgn(gz1["bat_speed"]["woba"]["cold"]), "e_review.E2_green_zwiebel_20_shuffles.bat_speed.woba.cold (one shuffle)")
put("gz1_bs_cold_woba_se", num(gz1["bat_speed"]["woba"]["cold_se"]), "e_review.E2 ... cold_se (one shuffle)")
G = lambda d, o: gz[d][o]
ob_raw_gap = G("onb", "onb")["hot_raw"] - G("onb", "onb")["cold_raw"]
ob_cor_gap = G("onb", "onb")["hot"] - G("onb", "onb")["cold"]
hr_base = G("hr", "hr")["base_rate"]
hr_raw_rel = (G("hr", "hr")["hot_raw"] - G("hr", "hr")["cold_raw"]) / (hr_base + G("hr", "hr")["cold_raw"])
hr_cor_rel = (G("hr", "hr")["hot"] - G("hr", "hr")["cold"]) / (hr_base + G("hr", "hr")["cold"])
put("gz_ob_gap_raw", num(ob_raw_gap), "e_review.E2_green_zwiebel_20_shuffles.onb.onb hot_raw - cold_raw (OBP points)")
put("gz_ob_gap_cor", num(ob_cor_gap), "e_review.E2_green_zwiebel_20_shuffles.onb.onb hot - cold")
put("gz_hr_rel_raw", pc(hr_raw_rel, 0), "e_review.E2 hr.hr (hot_raw - cold_raw) / (base + cold_raw)")
put("gz_hr_rel_cor", pc(hr_cor_rel, 0), "e_review.E2 hr.hr (hot - cold) / (base + cold)")
put("gz_hr_gap_raw", num(G("hr", "hr")["hot_raw"] - G("hr", "hr")["cold_raw"], 2), "e_review.E2 hr.hr hot_raw - cold_raw (pp)")
put("gz_hr_gap_cor", num(G("hr", "hr")["hot"] - G("hr", "hr")["cold"], 2), "e_review.E2 hr.hr hot - cold (pp)")
put("gz_hr_base", num(hr_base, 1), "e_review.E2 hr.hr base_rate (%)")
put("gz_bs_cold_woba", sgn(G("bat_speed", "woba")["cold"]), "e_review.E2_green_zwiebel_20_shuffles.bat_speed.woba.cold")
put("gz_bs_cold_woba_se", num(G("bat_speed", "woba")["cold_se"]), "e_review.E2 ... cold_se")
put("gz_bs_cold_hr", sgn(G("bat_speed", "hr")["cold"], 2), "e_review.E2_green_zwiebel_20_shuffles.bat_speed.hr.cold (pp)")
put("gz_bs_cold_hr_se", num(G("bat_speed", "hr")["cold_se"], 2), "e_review.E2 ... cold_se")
put("gz_bs_hot_hr", sgn(G("bat_speed", "hr")["hot"], 2), "e_review.E2 bat_speed.hr.hot (pp)")
put("gz_bs_hot_hr_se", num(G("bat_speed", "hr")["hot_se"], 2), "e_review.E2 bat_speed.hr.hot_se")
put("gz_bs_hot_woba", sgn(G("bat_speed", "woba")["hot"]), "e_review.E2 bat_speed.woba.hot")
put("gz_bs_hot_woba_se", num(G("bat_speed", "woba")["hot_se"]), "e_review.E2 bat_speed.woba.hot_se")
put("gz_woba_hot", sgn(G("woba", "woba")["hot"]), "e_review.E2_green_zwiebel_20_shuffles.woba.woba.hot")
put("gz_woba_hot_se", num(G("woba", "woba")["hot_se"]), "e_review.E2 ... hot_se")
put("gz_woba_hot_raw", sgn(G("woba", "woba")["hot_raw"]), "e_review.E2 ... hot_raw")
TREC = {"header": ["Hot or cold on", "Outcome over next 25 PA", "Hot, raw", "Hot, shuffle-corrected (SE)",
                 "Cold, raw", "Cold, shuffle-corrected (SE)"], "rows": []}
UNIT = {"woba": ("wOBA", 1), "onb": ("On-base rate", 1), "hr": ("Home-run rate (pp)", 2)}
DEF = {"woba": "wOBA", "onb": "On-base rate", "hr": "Home-run rate", "bat_speed": "Bat speed"}
for d in ("woba", "onb", "hr", "bat_speed"):
    for o in ("woba", "onb", "hr"):
        if d != "bat_speed" and o != d and not (d == "woba" and o == "hr"):
            continue
        v = gz[d][o]
        k = UNIT[o][1]
        TREC["rows"].append([DEF[d], UNIT[o][0], sgn(v["hot_raw"], k), f"{sgn(v['hot'], k)} {se_(v['hot_se'], k)}",
                           sgn(v["cold_raw"], k), f"{sgn(v['cold'], k)} {se_(v['cold_se'], k)}"])
TREC["albright"] = [["On-base sequence by PA", cnt(ob["n_hitter_seasons"]), num(ob["mean_z"], 2), num(ob["mean_z_game_shuffle"], 2),
                   f"{sgn(ob['mean_diff'], 2)} {se_(ob['se_diff'], 2)}"],
                  ["Bat speed above own median, by swing", cnt(bsr["n_hitter_seasons"]), num(bsr["mean_z"], 2),
                   num(bsr["mean_z_game_shuffle"], 2), f"{sgn(bsr['mean_diff'], 2)} {se_(bsr['se_diff'], 2)}"]]
TREC["albright_header"] = ["Runs test (Albright 1993)", "Hitter-seasons", "Mean z", "Mean z, game shuffle", "Difference (SE)"]

# ------------------------------------------------------------------ 4.2 value (Table 2)
forms = P["primary"]["forms"]
sw, rs = forms["swing_only"], forms["results_only"]
put("r2_swing", pct(sw["oos_r2_gain"], 2), "primary.forms.swing_only.oos_r2_gain")
put("r2_results", pct(rs["oos_r2_gain"], 2), "primary.forms.results_only.oos_r2_gain")
put("calib_swing", num(sw["calibration_slope"][0], 2), "primary.forms.swing_only.calibration_slope[0]")
put("calib_swing_se", num(sw["calibration_slope"][1], 2), "primary.forms.swing_only.calibration_slope[1]")
put("n_forecast", cnt(P["A2"]["primary"]["n"]), "A2.primary.n")
put("n_forecast_hitters", cnt(P["A2"]["primary"]["hitters"]), "A2.primary.hitters")
put("form_sd_pts", num(P["A6"]["form_sd_pts"]), "A6.form_sd_pts")
a2, a3 = P["A2"], P["A3"]
put("share_complete", pct(a2["ipw"]["share_complete"], 0), "A2.ipw.share_complete")
put("n_at_risk", cnt(a2["ipw"]["n_at_risk"]), "A2.ipw.n_at_risk")
put("n_crossing", cnt(a2["cross_season"]["n_crossing"]), "A2.cross_season.n_crossing")
put("n_regulars", cnt(a2["regulars_400pa"]["hitters"]), "A2.regulars_400pa.hitters")
put("ipw_w_lo", num(a2["ipw"]["weight_range"][0], 2), "A2.ipw.weight_range[0]")
put("ipw_w_hi", num(a2["ipw"]["weight_range"][1], 2), "A2.ipw.weight_range[1]")
put("reach_logit_zbs", num(a2["ipw"]["logit"]["z_bs"][0], 2), "A2.ipw.logit.z_bs[0]")
put("reach_logit_zbs_se", num(a2["ipw"]["logit"]["z_bs"][1], 2), "A2.ipw.logit.z_bs[1]")
put("reach_logit_zw", num(a2["ipw"]["logit"]["z_w"][0], 2), "A2.ipw.logit.z_w[0]")
put("reach_logit_zw_se", num(a2["ipw"]["logit"]["z_w"][1], 2), "A2.ipw.logit.z_w[1]")
proj_bot = [a3[s][b]["swing_only"]["bottom"] for s in ("primary_sample", "enlarged_sample")
            for b in ("marcel", "marcel_plus_history", "fangraphs_marcel", "fangraphs_plus_history")]
put("proj_bot_lo", num(-max(proj_bot)), "A3.*.{marcel,fangraphs}*.swing_only.bottom (max, sign flipped)")
put("proj_bot_hi", num(-min(proj_bot)), "A3.*.{marcel,fangraphs}*.swing_only.bottom (min, sign flipped)")
put("n_enlarged", cnt(a3["enlarged_sample"]["n"]), "A3.enlarged_sample.n")
put("n_enlarged_hitters", cnt(a3["enlarged_sample"]["hitters"]), "A3.enlarged_sample.hitters")

TFORM = {"header": ["Model", "Decision points", "Swing form, bottom decile", "Swing form, top decile", "wOBA streak, bottom decile",
                 "wOBA streak, top decile"], "rows": []}
dec = lambda d, k: f"{sgn(d['bump'][k])} {se_(d['se'][k])}"
TFORM["rows"].append(["Primary forecast (history, season to date, context)", cnt(a2["primary"]["n"]),
                   dec(sw["deciles"], 0), dec(sw["deciles"], 9), dec(rs["deciles"], 0), dec(rs["deciles"], 9)])
pf = forms["process"]["deciles"]
TFORM["rows"].append(["Primary forecast, all metrics except wOBA in the index", cnt(a2["primary"]["n"]), dec(pf, 0), dec(pf, 9), "", ""])
f2 = lambda r: [f"{sgn(r['bottom'])} {se_(r['bottom_se'])}", f"{sgn(r['top'])} {se_(r['top_se'])}"]
TFORM["rows"].append(["Survivorship: inverse-probability weights", cnt(a2["ipw"]["n_at_risk"]) + " at risk", *f2(a2["ipw"]), "", ""])
TFORM["rows"].append(["Survivorship: forward window may cross the season", cnt(a2["cross_season"]["n"]), *f2(a2["cross_season"]), "", ""])
TFORM["rows"].append([f"Survivorship: regulars ({DESIGN['a2_regular_pa']}+ PA the prior season)", cnt(a2["regulars_400pa"]["n"]),
                   *f2(a2["regulars_400pa"]), "", ""])
BL = {"marcel": "Marcel wOBA + season to date + context", "marcel_plus_history": "Marcel wOBA + full history",
      "fangraphs_marcel": "FanGraphs Marcel-style wRC+ + season to date + context",
      "fangraphs_plus_history": "FanGraphs Marcel-style wRC+ + full history"}
for s, lab in (("primary_sample", "primary sample"), ("enlarged_sample", "no history requirement")):
    for b, nm in BL.items():
        r = a3[s][b]
        TFORM["rows"].append([f"Projection baseline: {nm} ({lab})", cnt(a3[s]["n"]), *f2(r["swing_only"]), *f2(r["results_only"])])
put("t2_res_top_lo", A["res_top_lo"], "abstract.res_top_lo")

# ------------------------------------------------------------------ 4.3 platoon (Table 3) and B3
a7 = P["A7"]
B3 = J("b3.json")
pre, ref = a7["prespecified"], a7["refined_exploratory"]
bpre, bref = B3["rules"]["prespecified"], B3["rules"]["refined_exploratory"]
OB, OC0, OC = "option_b_vs_strict_platoon_runs_per50", "option_c0_vs_strict_platoon_runs_per50", "option_c_vs_strict_platoon_runs_per50"
for tag, a, b in (("pre", pre, bpre), ("ref", ref, bref)):
    put(f"a7_pairs_{tag}", cnt(a["n_pairs"]), f"A7.{tag}.n_pairs")
    put(f"a7_pairs_dec_{tag}", cnt(b["n_pairs"]), f"B3.{tag}.n_pairs (pairs with at least one decision)")
    put(f"a7_dec_{tag}", cnt(a["n_decisions"]), f"A7.{tag}.n_decisions")
    put(f"a7_ts_{tag}", cnt(a["n_team_seasons"]), f"A7.{tag}.n_team_seasons")
    put(f"a7_players_{tag}", cnt(a["n_players"]), f"A7.{tag}.n_players")
    put(f"a7_b_share_{tag}", pct(a[OB]["share_differs_from_platoon"], 0), f"A7.{tag}.{OB}.share_differs_from_platoon")
    put(f"a7_b_runs_{tag}", num(a[OB]["mean"], 2), f"A7.{tag}.{OB}.mean")
    put(f"a7_c0_share_{tag}", pct(a[OC0]["share_differs_from_platoon"], 0), f"A7.{tag}.{OC0}.share_differs_from_platoon")
    put(f"a7_c0_runs_{tag}", num(a[OC0]["mean"], 2), f"A7.{tag}.{OC0}.mean")
    put(f"a7_c_runs_{tag}", num(a[OC]["mean"], 2), f"A7.{tag}.{OC}.mean")
    put(f"a7_flip_{tag}", pct(a["share_form_flips_call"], 1), f"A7.{tag}.share_form_flips_call")
    put(f"a7_flip_runs_{tag}", num(a[OC]["mean"] - a[OC0]["mean"], 3), f"A7.{tag} option_c.mean - option_c0.mean")
    put(f"a7_real_{tag}", sgn(a["flipped_realised_minus_other_pts"][0]), f"A7.{tag}.flipped_realised_minus_other_pts[0]")
    put(f"a7_real_se_{tag}", num(a["flipped_realised_minus_other_pts"][1]), f"A7.{tag}.flipped_realised_minus_other_pts[1]")
    put(f"a7_pred_{tag}", sgn(a["flipped_predicted_form_gap_pts"]), f"A7.{tag}.flipped_predicted_form_gap_pts")
    put(f"a7_nflip_{tag}", cnt(a["n_flipped"]), f"A7.{tag}.n_flipped")
    g = a["mean_abs_gaps_pts"]
    put(f"a7_gap_form_{tag}", num(g["form"]), f"A7.{tag}.mean_abs_gaps_pts.form")
    put(f"a7_gap_talent_{tag}", num(g["talent"]), f"A7.{tag}.mean_abs_gaps_pts.talent")
    put(f"a7_gap_split_{tag}", num(g["split_term"]), f"A7.{tag}.mean_abs_gaps_pts.split_term")
    sec = b["secondary_own_vs_league_split"]
    put(f"b3_own_share_{tag}", pct(sec["share_calls_differ"], 1), f"B3.{tag}.secondary_own_vs_league_split.share_calls_differ")
    put(f"b3_own_runs_{tag}", num(sec["runs_per50_own_minus_league"], 3), f"B3.{tag}.secondary.runs_per50_own_minus_league")
    put(f"b3_c0L_runs_{tag}", num(sec["runs_per50_league_vs_platoon"], 2), f"B3.{tag}.secondary.runs_per50_league_vs_platoon")
    put(f"b3_pairs_cost_{tag}", pct(b["share_pairs_ride_hotter_costs_runs"], 0), f"B3.{tag}.share_pairs_ride_hotter_costs_runs")
    pp = b["per_pair_p10_p50_p90"]
    put(f"b3_pp_b_{tag}", f"{num(pp['runs_b'][1], 2)} ({num(pp['runs_b'][0], 2)} to {num(pp['runs_b'][2], 2)})", f"B3.{tag}.per_pair.runs_b")
    for h in ("L", "R"):
        hs = b["hitter_split_pts"][h]
        put(f"b3_split_{h}_mean_{tag}", num(hs["mean"]), f"B3.{tag}.hitter_split_pts.{h}.mean")
        put(f"b3_split_{h}_sd_{tag}", num(hs["sd"]), f"B3.{tag}.hitter_split_pts.{h}.sd")
        put(f"b3_split_{h}_n_{tag}", cnt(hs["n_hitters"]), f"B3.{tag}.hitter_split_pts.{h}.n_hitters")
        put(f"b3_split_{h}_p10_{tag}", num(hs["p10_p50_p90"][0]), f"B3.{tag}.hitter_split_pts.{h}.p10")
        put(f"b3_split_{h}_p90_{tag}", num(hs["p10_p50_p90"][2]), f"B3.{tag}.hitter_split_pts.{h}.p90")
put("relief_same_L", pct(pre["relief_same_hand_share"]["L"], 0), "A7.prespecified.relief_same_hand_share.L")
put("relief_same_R", pct(pre["relief_same_hand_share"]["R"], 0), "A7.prespecified.relief_same_hand_share.R")
pl = RA["platoon"]
put("ind_split_L", f"{num(pl['LHB_individual_p10_p50_p90'][0])}–{num(pl['LHB_individual_p10_p50_p90'][2])}", "results_adjusted.platoon.LHB_individual_p10_p50_p90")
put("ind_split_R", f"{num(pl['RHB_individual_p10_p50_p90'][0])}–{num(pl['RHB_individual_p10_p50_p90'][2])}", "results_adjusted.platoon.RHB_individual_p10_p50_p90")
put("extreme_vs_rhb", pct((float(A["top_swing"]) + float(A["bot_swing"])) / 2 / P["primary"]["platoon"]["RHB_league_split_pts"], 0),
    "abstract extreme / primary.platoon.RHB_league_split_pts")
put("extreme_vs_lhb", pct((float(A["top_swing"]) + float(A["bot_swing"])) / 2 / P["primary"]["platoon"]["LHB_league_split_pts"], 0),
    "abstract extreme / primary.platoon.LHB_league_split_pts")

TDEC = {"header": ["Rule", "Calls changed vs strict platoon", "Expected runs per 50 PA vs strict platoon",
                 "Per-pair median (p10 to p90)", "Calls changed (refined pairs, exploratory)", "Expected runs (refined pairs, exploratory)"],
      "rows": []}
ppr = lambda b, k: f"{num(b['per_pair_p10_p50_p90'][k][1], 2)} ({num(b['per_pair_p10_p50_p90'][k][0], 2)} to {num(b['per_pair_p10_p50_p90'][k][2], 2)})"
for lab, key, bk in (("(b) Start the hitter with the higher wOBA over his last 50 PA", OB, "runs_b"),
                     ("(c0) Talent and the hitter's own regressed split", OC0, "runs_c0"),
                     ("(c) Talent, own split and swing form", OC, "runs_c")):
    TDEC["rows"].append([lab, pct(pre[key]["share_differs_from_platoon"], 0), sgn(pre[key]["mean"], 2), ppr(bpre, bk),
                       pct(ref[key]["share_differs_from_platoon"], 0), sgn(ref[key]["mean"], 2)])
TDEC["rows"].append(["(c0L) Talent and the league split for the hand (secondary)", "", sgn(bpre["secondary_own_vs_league_split"]["runs_per50_league_vs_platoon"], 2),
                   "", "", sgn(bref["secondary_own_vs_league_split"]["runs_per50_league_vs_platoon"], 2)])
TDEC["rows"].append(["Form changes the call, (c) vs (c0)", pct(pre["share_form_flips_call"], 1), sgn(pre[OC]["mean"] - pre[OC0]["mean"], 3), "",
                   pct(ref["share_form_flips_call"], 1), sgn(ref[OC]["mean"] - ref[OC0]["mean"], 3)])
TDEC["rows"].append(["Own split changes the call, (c0) vs (c0L)", pct(bpre["secondary_own_vs_league_split"]["share_calls_differ"], 1),
                   sgn(bpre["secondary_own_vs_league_split"]["runs_per50_own_minus_league"], 3), "",
                   pct(bref["secondary_own_vs_league_split"]["share_calls_differ"], 1),
                   sgn(bref["secondary_own_vs_league_split"]["runs_per50_own_minus_league"], 3)])
TDEC["rows"].append(["Realised check on calls form changed (wOBA points, SE)",
                   f"{sgn(pre['flipped_realised_minus_other_pts'][0])} {se_(pre['flipped_realised_minus_other_pts'][1])}",
                   f"predicted {sgn(pre['flipped_predicted_form_gap_pts'])}", "",
                   f"{sgn(ref['flipped_realised_minus_other_pts'][0])} {se_(ref['flipped_realised_minus_other_pts'][1])}",
                   f"predicted {sgn(ref['flipped_predicted_form_gap_pts'])}"])

# ------------------------------------------------------------------ 4.4 decay, heterogeneity, A11
a1 = P["A1"]
fx = a1["horizons_fixed_sample"]
HW = P["horizon_weeks"]
for h in ("25", "50", "100", "150", "200"):
    put(f"days_{h}", f"{HW[h]['median_days']:.0f}", f"horizon_weeks.{h}.median_days")
    put(f"days_{h}_iqr", f"{HW[h]['iqr_days'][0]:.0f}–{HW[h]['iqr_days'][1]:.0f}", f"horizon_weeks.{h}.iqr_days")
    for m in ("bat_speed", "swing_len", "fast_swing", "z_swing", "woba"):
        put(f"fx_{m}_{h}", pct(fx[h][m]["persistence"]), f"A1.horizons_fixed_sample.{h}.{m}.persistence")
put("n_fixed", cnt(HW["n_decision_points"]), "horizon_weeks.n_decision_points")
put("n_fixed_bs", cnt(fx["200"]["bat_speed"]["n"]), "A1.horizons_fixed_sample.200.bat_speed.n")
cp = a1["changepoint"]
put("cp_before", pct(cp["before"]["persistence"]), "A1.changepoint.before.persistence")
put("cp_after", pct(cp["after"]["persistence"]), "A1.changepoint.after.persistence")
put("cp_after_se", num(100 * cp["after"]["se"]), "A1.changepoint.after.se")
put("cp_jump", pct(cp["share_jump_over_1mph"], 0), "A1.changepoint.share_jump_over_1mph")
put("cp_jump_sh", pct(cp["share_jump_over_1mph_shuffled"], 0), "A1.changepoint.share_jump_over_1mph_shuffled")
put("cp_med", num(cp["median_jump_mph"], 2), "A1.changepoint.median_jump_mph")
put("cp_med_sh", num(cp["median_jump_mph_shuffled"], 2), "A1.changepoint.median_jump_mph_shuffled")
put("cp_n", cnt(cp["n_hitter_seasons_fitted"]), "A1.changepoint.n_hitter_seasons_fitted")
ar = a1["ar1_game_level"]
ex_before = ar["raw"][0] - ar["raw_shuffled_mean"]
ex_after = ar["after_changepoint"][0] - ar["after_changepoint_shuffled_mean"]
put("ar_ex_before", num(ex_before, 3), "A1.ar1_game_level raw - raw_shuffled_mean")
put("ar_ex_after", num(ex_after, 3), "A1.ar1_game_level after - after_shuffled_mean")
put("ar_survive", pct(ex_after / ex_before, 0), "A1.ar1_game_level excess after / before")
bl = a1["block_lag1"]
put("blk_bs", sgn(bl["bat_speed"]["actual"], 2), "A1.block_lag1.bat_speed.actual")
put("blk_bs_sh", sgn(bl["bat_speed"]["shuffled_mean"], 2), "A1.block_lag1.bat_speed.shuffled_mean")
put("blk_w", sgn(bl["woba"]["actual"], 2), "A1.block_lag1.woba.actual")
put("blk_w_sh", sgn(bl["woba"]["shuffled_mean"], 2), "A1.block_lag1.woba.shuffled_mean")
a10 = P["A10"]
put("p_days14", pct(a10["days14_14"]["persistence"]["bat_speed"]["persistence"]), "A10.days14_14.persistence.bat_speed")
put("p_days28", pct(a10["days28_28"]["persistence"]["bat_speed"]["persistence"]), "A10.days28_28.persistence.bat_speed")
put("p_w100", pct(a10["w100_100"]["persistence"]["bat_speed"]["persistence"]), "A10.w100_100.persistence.bat_speed")
a4 = P["A4"]["bat_speed"]
s50, s100 = a4["step50"]["actual"], a4["step100"]
put("a4_hitters", cnt(s50["hitters"]), "A4.bat_speed.step50.actual.hitters")
put("a4_pairs", f"{s50['median_pairs']:.0f}", "A4.bat_speed.step50.actual.median_pairs")
put("a4_tau", num(s50["tau_dl"], 2), "A4.bat_speed.step50.actual.tau_dl")
put("a4_tau_mm", num(s50["mixed_model_slope_sd"], 2), "A4.bat_speed.step50.actual.mixed_model_slope_sd")
put("a4_rel", num(s50["reliability_median_hitter"], 2), "A4.bat_speed.step50.actual.reliability_median_hitter")
put("a4_cred", cnt(s50["n_credibly_different"]), "A4.bat_speed.step50.actual.n_credibly_different")
put("a4_tau_null", num(a4["step50"]["null"]["tau_dl"], 2), "A4.bat_speed.step50.null.tau_dl")
put("a4_tau100", num(s100["actual"]["tau_dl"], 2), "A4.bat_speed.step100.actual.tau_dl")
put("a4_tau100_null", num(s100["null"]["tau_dl"], 2), "A4.bat_speed.step100.null.tau_dl")
put("a4_hitters100", cnt(s100["actual"]["hitters"]), "A4.bat_speed.step100.actual.hitters")
w4 = P["A4"]["woba"]["step50"]["actual"]
put("a4_tau_woba", num(w4["tau_dl"], 2), "A4.woba.step50.actual.tau_dl")
put("a4_tau_woba_null", num(P["A4"]["woba"]["step50"]["null"]["tau_dl"], 2), "A4.woba.step50.null.tau_dl")
put("a4_cred_woba", cnt(w4["n_credibly_different"]), "A4.woba.step50.actual.n_credibly_different")

a11 = J("a11.json")
r11 = a11["real"]
A11 = P["A11"]
put("a11_b", num(A11["b"], 2), "A11.b")
put("a11_b_null", num(A11["b_null"], 2), "A11.b_null")
put("a11_excess", sgn(A11["b_excess"], 2), "A11.b_excess")
put("a11_excess_se", num(A11["b_excess_se"], 2), "A11.b_excess_se")
put("a11_n", cnt(A11["b_n"]), "A11.b_n")
put("a11_ex_2425", sgn(r11["A11b"]["pair_2024_2025"]["excess"], 2), "a11.real.A11b.pair_2024_2025.excess")
put("a11_ex_2526", sgn(r11["A11b"]["pair_2025_2026"]["excess"], 2), "a11.real.A11b.pair_2025_2026.excess")
put("a11_ex_ipw", sgn(r11["A11b"]["ipw"]["excess"], 2), "a11.real.A11b.ipw.excess")
put("a11_rnext", num(A11["R_next"], 2), "A11.R_next")
ci = a11["decision"]["primary"]["ci_R_next"]
put("a11_rnext_ci", f"{num(ci[0], 2)}–{num(ci[1], 2)}", "a11.decision.primary.ci_R_next")
cib = a11["decision"]["primary"]["ci_b"]
put("a11_b_ci", f"{num(cib[0], 2)}–{num(cib[1], 2)}", "a11.decision.primary.ci_b")
sig = A11["signature"]
put("a11_r1", num(sig["real"]["R_within_1"], 2), "A11.signature.real.R_within_1")
put("a11_r4", num(sig["real"]["R_within_4"], 2), "A11.signature.real.R_within_4")
put("a11_r1_none", num(sig["a_none"]["R_within_1"]["mean"], 2), "A11.signature.a_none.R_within_1.mean")
put("a11_r4_none", num(sig["a_none"]["R_within_4"]["mean"], 2), "A11.signature.a_none.R_within_4.mean")
put("a11_rnext_none", num(sig["a_none"]["R_next"]["mean"], 2), "A11.signature.a_none.R_next.mean")
put("a11_b_step", num(sig["c_step"]["b"]["mean"], 2), "A11.signature.c_step.b.mean")
put("a11_rnext_step", num(sig["c_step"]["R_next"]["mean"], 2), "A11.signature.c_step.R_next.mean")
put("a11_b_mix", num(sig["d_mix"]["b"]["mean"], 2), "A11.signature.d_mix.b.mean")
put("a11_rnext_mix", num(sig["d_mix"]["R_next"]["mean"], 2), "A11.signature.d_mix.R_next.mean")
put("a11_hl", f"{A11['half_life_pa']:.0f}", "A11.half_life_pa")
reps = a11["sim_replicates"]["a_none"]
put("a11_reps", f"{len(reps)}", "a11.sim_replicates.a_none (count)")
put("a11_plan", A11["plan_sha16"], "A11.plan_sha16")

# ------------------------------------------------------------------ 4.5 pitchers
a5 = P["A5"]
pp5 = a5["persistence"]
for m in pp5:
    put(f"p5_{m}", pct(pp5[m]["persistence"]), f"A5.persistence.{m}.persistence")
    put(f"p5_{m}_se", num(100 * pp5[m]["se"]), f"A5.persistence.{m}.se")
put("a5_points", cnt(a5["n_points"]), "A5.n_points")
put("a5_pitchers", cnt(a5["n_pitchers"]), "A5.n_pitchers")
cx = a5["cross_xwoba_ag"]
for m in ("ff_velo", "ff_ext", "ff_spin", "ff_relz", "woba_ag", "xwoba_ag", "kbb"):
    put(f"cx_{m}", sgn(cx[m]["pts_per_sd"]), f"A5.cross_xwoba_ag.{m}.pts_per_sd")
    put(f"cx_{m}_se", num(cx[m]["se"]), f"A5.cross_xwoba_ag.{m}.se")
st = a5["starters_results_vs_velocity"]
put("st_n", cnt(st["actual"]["n"]), "A5.starters_results_vs_velocity.actual.n")

# ------------------------------------------------------------------ robustness, IL, postseason
a8 = P["A8"]
put("il_n_points", cnt(a8["n_points"]), "A8.n_points")
put("il_placements", cnt(a8["n_placements_in_window"]), "A8.n_placements_in_window")
put("il_base", pct(a8["base_rate"]), "A8.base_rate")
fam = a8["risk"]
for f_ in ("oblique", "hamstring", "hand_wrist"):
    put(f"il_{f_}", f"{num(fam[f_]['OR_per_minus1SD'], 2)} ({num(fam[f_]['ci'][0], 2)}–{num(fam[f_]['ci'][1], 2)})", f"A8.risk.{f_}")
put("il_quad_p", num(a8["quadratic_prespecified_secondary"]["p_z2"], 2), "A8.quadratic_prespecified_secondary.p_z2")
put("il_dec_lo", pct(min(a8["il_rate_by_decile"]["il_rate"])), "A8.il_rate_by_decile.il_rate (min)")
put("il_dec_hi", pct(max(a8["il_rate_by_decile"]["il_rate"])), "A8.il_rate_by_decile.il_rate (max)")
pg = a8["persistence_by_group"]
put("il_pers_all", pct(pg["all"]["persistence"]), "A8.persistence_by_group.all.persistence")
put("il_pers_clean", pct(pg["no_IL_next30_no_recent_return"]["persistence"]), "A8.persistence_by_group.no_IL_next30_no_recent_return.persistence")
put("il_sd_mph", num(a8["sd_bs_dev_mph"], 2), "A8.sd_bs_dev_mph")
a6 = P["A6"]
put("a6_n", cnt(a6["n_hitter_postseasons"]), "A6.n_hitter_postseasons")
put("a6_pa", cnt(a6["total_pa"]), "A6.total_pa")
put("a6_b", sgn(a6["pts_per_sd_form"]), "A6.pts_per_sd_form")
put("a6_se", num(a6["se"]), "A6.se")
put("a6_ci", f"{num(a6['ci95'][0])} to {num(a6['ci95'][1])}", "A6.ci95")
put("a6_mde", num(a6["mde_80pct_power"], 0), "A6.mde_80pct_power")
put("a6_expect", num(a6["regular_season_pts_per_sd_form"]), "A6.regular_season_pts_per_sd_form")
for k in ("competitive_filter_off", "park_week_adjustment_off", "drop_first_30_days"):
    put(f"rb_{k}_bs", pct(a10[k]["persistence"]["bat_speed"]["persistence"]), f"A10.{k}.persistence.bat_speed")
    put(f"rb_{k}_sl", pct(a10[k]["persistence"]["swing_len"]["persistence"]), f"A10.{k}.persistence.swing_len")
    put(f"rb_{k}_fs", pct(a10[k]["persistence"]["fast_swing"]["persistence"]), f"A10.{k}.persistence.fast_swing")
for y in ("2024", "2025", "2026"):
    s_ = a10[f"season_{y}"]
    put(f"rb_{y}_bs", pct(s_["persistence"]["bat_speed"]["persistence"]), f"A10.season_{y}.persistence.bat_speed")
    put(f"rb_{y}_bot", sgn(s_["swing_form"]["bottom"]), f"A10.season_{y}.swing_form.bottom")
    put(f"rb_{y}_bot_se", num(s_["swing_form"]["bottom_se"]), f"A10.season_{y}.swing_form.bottom_se")
    put(f"rb_{y}_top", sgn(s_["swing_form"]["top"]), f"A10.season_{y}.swing_form.top")
bys = [a10[f"season_{y}"]["persistence"]["bat_speed"]["persistence"] for y in (2024, 2025, 2026)]
put("rb_season_bs_lo", pc(min(bys), 0), "A10.season_*.persistence.bat_speed (min)")
put("rb_season_bs_hi", pct(max(bys), 0), "A10.season_*.persistence.bat_speed (max)")

# Appendix Table A1 (A10, A2, A3, windows)
TA1 = {"header": ["Specification", "Bat speed", "Swing length", "wOBA", "Swing form, bottom decile (SE)", "Swing form, top decile (SE)"],
       "rows": []}
pr = lambda d, m: pct(d[m]["persistence"]) if m in d else ""
f2s = lambda r: [f"{sgn(r['bottom'])} {se_(r['bottom_se'])}", f"{sgn(r['top'])} {se_(r['top_se'])}"]
pp0 = a10["primary_50_50"]
TA1["rows"].append([f"Primary, 50 PA → 50 PA ({DESIGN['a10_shuffles']}-shuffle null)", pr(pp0["persistence"], "bat_speed"),
                    pr(pp0["persistence"], "swing_len"), pr(pp0["persistence"], "woba"), *f2s(pp0["swing_form"])])
for k, lab in (("rec25_fwd25", "25 PA → 25 PA (single shuffle)"), ("rec50_fwd25", "50 PA → 25 PA (single shuffle)"),
               ("rec100_fwd50", "100 PA → 50 PA (single shuffle)")):
    w_ = {r["metric"]: r for r in RA[k]["persistence"]}
    fo = RA[k]["woba_forms"]["swing_only"]["deciles"]
    TA1["rows"].append([lab, pr(w_, "bat_speed"), pr(w_, "swing_len"), pr(w_, "woba"), dec(fo, 0), dec(fo, 9)])
for k, lab in (("w100_100", "100 PA → 100 PA"), ("days14_14", "14 days → 14 days"), ("days28_28", "28 days → 28 days"),
               ("park_week_adjustment_off", "Park and week adjustment off"), ("competitive_filter_off", "Competitive-swing filter off"),
               ("drop_first_30_days", "First 30 days of each season dropped")):
    s_ = a10[k]
    TA1["rows"].append([lab, pr(s_["persistence"], "bat_speed"), pr(s_["persistence"], "swing_len"), pr(s_["persistence"], "woba"),
                        *(f2s(s_["swing_form"]) if "swing_form" in s_ else ["", ""])])
for y in ("2024", "2025", "2026"):
    s_ = a10[f"season_{y}"]
    lab = f"{y} only" + (" (flag: see note)" if y == "2026" else "")
    TA1["rows"].append([lab, pr(s_["persistence"], "bat_speed"), pr(s_["persistence"], "swing_len"), pr(s_["persistence"], "woba"),
                        *f2s(s_["swing_form"])])
for k, lab in (("ipw", "Survivorship: inverse-probability weights"), ("cross_season", "Survivorship: window crosses the season"),
               ("regulars_400pa", "Survivorship: regulars only")):
    TA1["rows"].append([lab, "", "", "", *f2s(a2[k])])
for b, nm in BL.items():
    TA1["rows"].append([f"Projection: {nm}", "", "", "", *f2s(a3["primary_sample"][b]["swing_only"])])

# Appendix Table A2 (A11 signature)
WORLDS = [("real", "Real"), ("a_none", "(a) No state"), ("b_hl50", "(b) AR, half-life 50 PA"), ("b_hl150", "(b) AR, 150 PA"),
          ("b_hl400", "(b) AR, 400 PA"), ("c_step", "(c) Lasting step"), ("d_mix", "(d) Mix")]
SIGROWS = [("p5050", "50 → 50 persistence"), ("decay_25", "Fixed sample, next 25 PA"), ("decay_100", "Fixed sample, next 100 PA"),
           ("decay_200", "Fixed sample, next 200 PA"), ("share_jump_over_1mph", "Changepoint jump over 1 mph"),
           ("R_within_1", "Held-out step retention, block 1"), ("R_within_4", "Held-out step retention, block 4"),
           ("R_next", "Step retention into next season"), ("b", "Offseason carry-over slope b"),
           ("b_null", "Shuffled-null slope"), ("b_excess", "b minus null")]
TA2 = {"header": ["Signature"] + [w[1] for w in WORLDS], "rows": []}
for k, lab in SIGROWS:
    row = [lab]
    for w, _ in WORLDS:
        v = sig[w][k]
        row.append(num(v, 2) if w == "real" else f"{num(v['mean'], 2)} ({num(v['sd'], 2)})")
    TA2["rows"].append(row)

# ------------------------------------------------------------------ simulation
SIM = P["simulation"]
put("sim_hot_bs_se", num(100 * SIM["hot"]["bat_speed"]["se"]), "simulation.hot.bat_speed.se")
put("sim_null_bs_se", num(100 * SIM["null"]["bat_speed"]["se"]), "simulation.null.bat_speed.se")
put("sim_hot_w", pct(SIM["hot"]["woba"]["persistence"]), "simulation.hot.woba.persistence")
put("sim_null_w", pct(SIM["null"]["woba"]["persistence"]), "simulation.null.woba.persistence")

# ------------------------------------------------------------------ companion (B2)
comp_path = sys.argv[2] if len(sys.argv) > 2 else None
if comp_path:
    C_ = json.load(open(comp_path))
    e = C_["entries"]
    get = lambda key: e[key]["value"]
    base = "twin.r1.public.c_OR_split_on_I"
    COMP = {"or": get(f"{base}.OR_per_minus1SD"), "lo": get(f"{base}.lo"), "hi": get(f"{base}.hi"),
            "events": get(f"{base}.events"), "key": base, "file_sha16": sha16(comp_path),
            "title": "Can Ball-Tracking Predict Injuries in the MLB Player?", "status": "unpublished companion work (SSAC27 abstract, 2026)"}
else:
    COMP = P.get("paper", {}).get("companion")
    if COMP is None:
        sys.exit("no companion snapshot: pass the companion synthesis_results.json once")
put("comp_or", num(COMP["or"], 2), f"companion.{COMP['key']}.OR_per_minus1SD")
put("comp_lo", num(COMP["lo"], 2), f"companion.{COMP['key']}.lo")
put("comp_hi", num(COMP["hi"], 2), f"companion.{COMP['key']}.hi")
put("comp_events", cnt(COMP["events"]), f"companion.{COMP['key']}.events")

with open(os.path.join(R, "a7_pairs.csv")) as f:
    put("a7_csv_rows", cnt(sum(1 for _ in csv.reader(f)) - 1), "a7_pairs.csv rows")

# ------------------------------------------------------------------ Reviewer 2 additions (plan addendum E, exploratory)
import math  # noqa: E402

wp = {r["metric"]: r for r in RA["rec50_fwd50"]["persistence"]}
sdw, sdb = 1000 * wp["woba"]["sd_recent_dev"], wp["bat_speed"]["sd_recent_dev"]
p_up = PERS["woba"]["persistence"] + 1.96 * PERS["woba"]["se"]
put("sd_rec_woba", num(sdw, 0), "results_adjusted.rec50_fwd50.persistence[woba].sd_recent_dev x 1000")
put("woba_state_sd", num(math.sqrt(p_up * sdw ** 2), 0), "sqrt(woba persistence upper 95% x var(recent wOBA deviation)), plan addendum E1")
put("bs_state_sd", num(math.sqrt(PERS["bat_speed"]["persistence"] * sdb ** 2), 2), "sqrt(bat-speed persistence x var(recent deviation)), E1")
Hs = [25, 50, 100, 150, 200]
Pv = {h: fx[str(h)]["bat_speed"]["persistence"] for h in Hs}
for lo_, hi_ in zip(Hs[:-1], Hs[1:]):
    put(f"blk_bs_{lo_ + 1}_{hi_}", pct((hi_ * Pv[hi_] - lo_ * Pv[lo_]) / (hi_ - lo_)), f"A1 fixed-sample block increment ({lo_},{hi_}], E1")


def cochran(est):
    w = [1 / se ** 2 for _, se in est]
    m_ = sum(wi * b for wi, (b, _) in zip(w, est)) / sum(w)
    q_ = sum(wi * (b - m_) ** 2 for wi, (b, _) in zip(w, est))
    return q_, math.exp(-q_ / 2)          # chi-square survival, 2 degrees of freedom


qb = cochran([(a10[f"season_{y}"]["swing_form"]["bottom"], a10[f"season_{y}"]["swing_form"]["bottom_se"]) for y in (2024, 2025, 2026)])
qt = cochran([(a10[f"season_{y}"]["swing_form"]["top"], a10[f"season_{y}"]["swing_form"]["top_se"]) for y in (2024, 2025, 2026)])
put("q_bot", num(qb[0]), "Cochran Q, A10 season swing-form bottom deciles, E1")
put("q_bot_p", num(qb[1], 3), "Cochran Q p (df 2)")
put("q_top", num(qt[0]), "Cochran Q, A10 season swing-form top deciles, E1")
put("q_top_p", num(qt[1], 3), "Cochran Q p (df 2)")
A3B = lambda s_: [b for b in a3[s_] if isinstance(a3[s_][b], dict) and "results_only" in a3[s_][b]]
SPECS = ("w100_100", "days14_14", "days28_28", "competitive_filter_off", "park_week_adjustment_off")
rt = [(rs["deciles"]["bump"][9], rs["deciles"]["se"][9])] + [(a3[s_][b]["results_only"]["top"], a3[s_][b]["results_only"]["top_se"])
                                                              for s_ in a3 for b in A3B(s_)]
rt += [(a10[k]["results_form"]["top"], a10[k]["results_form"]["top_se"]) for k in SPECS]
put("res_top_n", cnt(len(rt)), "wOBA-streak top decile specifications (as abstract.res_top_lo/hi)")
put("res_top_pos", cnt(sum(v > 0 for v, _ in rt)), "count positive, E1")
put("res_top_2se", cnt(sum(v / se >= 2 for v, se in rt)), "count >= 2 SE, E1")
stp = [sw["deciles"]["bump"][9], a2["ipw"]["top"], a2["cross_season"]["top"], a2["regulars_400pa"]["top"]]
stp += [a3[s_][b]["swing_only"]["top"] for s_ in a3 for b in A3B(s_)]
stp += [a10[k]["swing_form"]["top"] for k in SPECS]
put("swing_top_lo", num(min(stp)), "swing-form top decile over the abstract's specification set (min)")
put("swing_top_hi", num(max(stp)), "swing-form top decile over the abstract's specification set (max)")
for k, tag in (("rec25_fwd25", "w2525"), ("rec50_fwd25", "w5025")):
    d_ = RA[k]["woba_forms"]["swing_only"]["deciles"]
    put(f"{tag}_bot", sgn(d_["bump"][0]), f"results_adjusted.{k}.woba_forms.swing_only.deciles.bump[0]")
    put(f"{tag}_bot_se", num(d_["se"][0]), f"results_adjusted.{k}.woba_forms.swing_only.deciles.se[0]")
put("velo_sd_mph2", num(st["actual"]["sd_velo_dev_mph"], 2), "A5.starters_results_vs_velocity.actual.sd_velo_dev_mph")
put("il_pers_before", pct(pg["IL_next30"]["persistence"]), "A8.persistence_by_group.IL_next30.persistence")
put("il_pers_before_se", num(100 * pg["IL_next30"]["se"]), "A8.persistence_by_group.IL_next30.se")
put("il_pers_clean_se", num(100 * pg["no_IL_next30_no_recent_return"]["se"]), "A8.persistence_by_group.no_IL_next30_no_recent_return.se")
fo = a10["competitive_filter_off"]["swing_form"]
put("rb_filter_off_bot", sgn(fo["bottom"]), "A10.competitive_filter_off.swing_form.bottom")
put("rb_filter_off_top", sgn(fo["top"]), "A10.competitive_filter_off.swing_form.top")
put("pers_a10_bs", pct(a10["primary_50_50"]["persistence"]["bat_speed"]["persistence"]), "A10.primary_50_50.persistence.bat_speed")
for tag, a in (("pre", pre), ("ref", ref)):
    put(f"a7_b_cost_{tag}", num(-a[OB]["mean"], 2), f"A7.{tag}.{OB}.mean (sign flipped)")
E3 = EV["E3_lineups"]
for tag, key in (("pre", "prespecified"), ("ref", "refined_exploratory")):
    e = E3[key]
    put(f"e3_b_n_{tag}", cnt(e["b_changed_n"]), f"e_review.E3_lineups.{key}.b_changed_n")
    put(f"e3_b_real_{tag}", sgn(e["b_realised_minus_platoon_pick_pts"][0]), f"e_review.E3_lineups.{key}.b_realised_minus_platoon_pick_pts[0]")
    put(f"e3_b_real_se_{tag}", num(e["b_realised_minus_platoon_pick_pts"][1]), f"e_review.E3_lineups.{key}.b_realised_minus_platoon_pick_pts[1]")
    put(f"e3_top_share_{tag}", pct(e["largest_team_season_share"], 0), f"e_review.E3_lineups.{key}.largest_team_season_share")
    put(f"e3_max_pairs_{tag}", cnt(e["max_pairs_per_hitter"]), f"e_review.E3_lineups.{key}.max_pairs_per_hitter")
WN = {"a_none": "the world with no state", "b_hl50": "the 50-PA autoregressive world", "b_hl150": "the 150-PA autoregressive world",
      "b_hl400": "the 400-PA autoregressive world", "c_step": "the lasting-step world", "d_mix": "the mixed world"}
put("a11_closest", WN[a11["decision"]["primary"]["closest_world"]], "a11.decision.primary.closest_world")

P["paper"] = {"meta": {"plan": "PLAN_ADDENDUM_2026-10-02d.md", "abstract_sha16": ABSTRACT_SHA16,
                       "inputs_sha16": {n: sha16(os.path.join(R, n)) for n in ("a11.json", "b3.json", "a7_pairs.csv", "e_review.json",
                                                                                "results_adjusted.json", "paper_counts.json")},
                       "plan_review": "PLAN_ADDENDUM_2026-10-02e.md (exploratory: E1 derived quantities, E2 20-shuffle Green-Zwiebel, E3 lineup checks)",
                       "exploratory": ["A7 refined pairing rule", "A11 (prespecified rule returned undetermined)", "plan addendum E (E1-E3)",
                                       "A1 non-overlapping block autocorrelation", "A5 four-seam arm angle"]},
              "text": T, "source": SRC, "design": DESIGN,
              "tables": {"T1": T1, "T2": TREC, "T3": TFORM, "T4": TDEC, "TA1": TA1, "TA2": TA2},
              "companion": COMP}
P["B3"] = B3
json.dump(P, open(PR_PATH, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(f"paper block: {len(T)} text values, {sum(len(t['rows']) for t in P['paper']['tables'].values())} table rows")
