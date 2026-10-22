"""Compare the analyses on IL stints (author decision of 2026-10-20) with those on IL placement records.

python3 ojsm_stints_compare.py RESULTS_DIR STINTS_DIR OUT_JSON [RECORD_STINT_FILE CORRECTED_STINT_FILE]

The optional file names (2026-10-21, for the final run of PLAN_ADDENDUM_2026-10-17h.md item 14) are the stint files of STINTS_DIR
in repo/data_public/; they default to the stint files of 2026-10-20.

Checks that the season-level results are identical except where a merged record named another injury family (their
outcomes are placements of a family in a season), that no hitter-season changes oblique status (so the oblique-only analyses
that were not rerun, ajsm_f1_post_hoc.json, cannot change), and records how the descriptive counts (F3) and the game-level
estimates moved.
"""
import json
import os
import sys

RES, ST, OUT = sys.argv[1:4]
STF = sys.argv[4:6] if len(sys.argv) > 5 else ["il_transactions_hitters_2024_2026_stints.txt", "il_transactions_hitters_2024_2026_corrected_stints.txt"]
J = lambda d, f: json.load(open(os.path.join(d, f)))
out = {}
# hitter-season families on records and on stints: merging changes a hitter-season only through the family of a merged record
def fams(path):
    o = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        b, ev = line.split(":")
        for e in ev.split():
            if e[6] == "P":
                o.setdefault((int(b), 2000 + int(e[:2])), set()).add(e[7])
    return o
DP = os.path.join(os.path.dirname(os.path.abspath(RES)), "data_public")
for lab, f0, f1 in (("record-date file", "il_transactions_hitters_2024_2026.txt", STF[0]),
                    ("corrected file", "il_transactions_hitters_2024_2026_corrected.txt", STF[1])):
    a_, b_ = fams(os.path.join(DP, f0)), fams(os.path.join(DP, f1))
    ch = [{"batter": k[0], "season": k[1], "records": sorted(a_[k]), "stints": sorted(b_.get(k, set()))} for k in sorted(a_) if a_[k] != b_.get(k, set())]
    out[f"family changes, {lab}"] = ch
    assert all("O" not in c_["records"] and "O" not in c_["stints"] for c_ in ch), ch     # no oblique status changes
# the season-level models: the any, oblique, hamstring and hand/wrist outcomes and F2 must be identical; the 'other' outcome of
# the record-date file may change through a merged record of another family
def same(x, y):
    return json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True)
for f in ("ajsm_season.json", "ajsm_season_corrected.json"):
    a, b = J(RES, f), J(ST, f)
    assert same(a["F2"], b["F2"]), (f, "F2")
    for y in ("y_any", "y_obl", "y_ham", "y_hw"):
        assert same(a["F1"]["primary"][y], b["F1"]["primary"][y]) and a["F1"]["cohort"]["events"][y] == b["F1"]["cohort"]["events"][y], (f, y)
    out[f"{f}: F2 and the any, oblique, hamstring and hand/wrist F1 results identical"] = True
    out[f"{f}: F1 other-family events"] = {"records": a["F1"]["cohort"]["events"]["y_oth"], "stints": b["F1"]["cohort"]["events"]["y_oth"]}
    out[f"{f}: F1 other-family model identical"] = same(a["F1"]["primary"]["y_oth"], b["F1"]["primary"]["y_oth"])
a, b = J(RES, "ajsm_season_corrected.json")["F3"], J(ST, "ajsm_season_corrected.json")["F3"]
out["F3"] = {k: {"records": a[k], "stints": b[k]} for k in ("placements", "hitters_with_placement", "cohort_hitters", "by_family",
                                                             "per_1000_PA_by_season", "season_ending_share_all", "stint_days_all",
                                                             "repeat_same_family_365d")}
ga, gb = J(RES, "ajsm_game_ojsm.json"), J(ST, "ajsm_game_ojsm.json")
w = lambda g, f: g["F4a"]["windows"][f]["1-14"]
out["F4a_1-14"] = {f: {"records": [w(ga, f)["diff"], w(ga, f)["n_inj"]], "stints": [w(gb, f)["diff"], w(gb, f)["n_inj"]]}
                   for f in ("all", "oblique", "hamstring", "hand_wrist", "other")}
out["F4b_stints"] = {"records": ga["F4b"]["n_stints"], "stints": gb["F4b"]["n_stints"]}
dif, rows = [], {}
for f in ("all", "oblique", "hamstring", "hand_wrist", "other"):
    for wn in ("1-10", "21-40", "41-60"):
        x, y = ga["F4b"]["effects_all_controls"][f]["bat_speed"][wn], gb["F4b"]["effects_all_controls"][f]["bat_speed"][wn]
        rows[f"{f} {wn}"] = {"records": [x["mean"], x["n"]], "stints": [y["mean"], y["n"]]}
        dif.append(abs(x["mean"] - y["mean"]))
out["F4b_every_eligible_control"] = rows
out["F4b_every_eligible_control_max_abs_change"] = max(dif)
rows5 = {}
for f in ("all", "oblique", "hamstring", "hand_wrist", "other"):
    for wn in ("1-10", "21-40", "41-60"):
        x, y = ga["F4b"]["effects"][f]["bat_speed"][wn], gb["F4b"]["effects"][f]["bat_speed"][wn]
        rows5[f"{f} {wn}"] = {"records": [x["mean"], x["n"]], "stints": [y["mean"], y["n"]]}
out["F4b_5_controls"] = rows5
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("F4b_every_eligible_control", "F4b_5_controls")}, indent=1))
print("max abs change, every eligible control:", out["F4b_every_eligible_control_max_abs_change"])
