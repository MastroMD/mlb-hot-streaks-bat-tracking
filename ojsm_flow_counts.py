"""Post hoc, added after the cold read of 2026-10-17 (PLAN_ADDENDUM_2026-10-17h.md, item 10): the exclusion steps of the
before-placement and after-return samples for the flow diagram (Figure 1), and the reasons the IL check found placements
missing from the record-date file (Supplemental Table S4).

python3 ojsm_flow_counts.py DATA_DIR IL_FILE CODED_CSV RAW_DESCRIPTIONS ORIGINAL_IL_FILE OUT_JSON

Runs the data and baseline code of ajsm_game_ojsm.py (up to the at-risk models, which are not needed) and its post_rows
function, and counts each placement once at the first rule it fails, in the order of the code. Checks that the final
counts equal the published ones (568 and 547 placements before placement; 466 stints after return).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, ILF, CODED, RAW, ILO, OUT = sys.argv[1:7]
src = open(os.path.join(HERE, "ajsm_game_ojsm.py"), encoding="utf-8").read()
head = src[:src.index("# ------------------------------------------------------------------ F4a at-risk A8 and F4e")]
f4b = src[src.index("# ------------------------------------------------------------------ F4b return to performance"):src.index("stints = []")]
RUN = os.path.join(os.path.dirname(os.path.abspath(OUT)), "flow_counts_run")   # the reused code writes its F4a rows here
os.makedirs(RUN, exist_ok=True)
sys.argv = [os.path.join(HERE, "ajsm_game_ojsm.py"), DATA, ILF, os.path.join(RUN, "unused.json"), "--shuffles", "5", "--coded", CODED]
NS = {"__name__": "__main__", "__file__": os.path.join(HERE, "ajsm_game_ojsm.py")}
exec(compile(head + "\n" + f4b, "ajsm_game_ojsm.py (data, F4a windows, post_rows)", "exec"), NS)
P, G, LABEL_END, baseline, pre_rows, post_rows = (NS[k] for k in ("P", "G", "LABEL_END", "baseline", "pre_rows", "post_rows"))
G_json = json.load(open(os.path.join(os.path.dirname(os.path.abspath(OUT)), "ajsm_game_ojsm.json")))   # 2026-10-20: the run beside OUT_JSON

# ------------------------------------------------------------------ before placement (F4a): the order of pre_rows
pre = {"placements": int(len(P))}
c = {"after_label_end_or_no_games_that_season": 0, "no_baseline": 0, "no_window_with_5_competitive_swings": 0,
     "with_rows": 0, "days_1_14_with_20_competitive_swings": 0}
for p in P.itertuples():
    y = p.start.year
    if (p.batter, y) not in G or p.start > LABEL_END:
        c["after_label_end_or_no_games_that_season"] += 1; continue
    base, _ = baseline(p.batter, y, p.start - pd.Timedelta(days=30))
    if base is None:
        c["no_baseline"] += 1; continue
    r = pre_rows(p.batter, y, p.start, 1, p.fam, p.imp)
    if not r:
        c["no_window_with_5_competitive_swings"] += 1; continue
    c["with_rows"] += 1
    c["days_1_14_with_20_competitive_swings"] += any(x["win"] == "1-14" for x in r)
pre.update(c)
assert pre["with_rows"] == G_json["F4a"]["n_placements_with_rows"], pre
assert pre["days_1_14_with_20_competitive_swings"] == G_json["F4a"]["windows"]["all"]["1-14"]["n_inj"], pre

# ------------------------------------------------------------------ after return (F4b): the order of the stint loop
post = {"placements": int(len(P))}
c = {"no_activation_in_the_same_season": 0, "no_games_that_season": 0, "no_baseline": 0,
     "no_game_after_return_before_a_later_placement": 0, "stints": 0}
for p in P.itertuples():
    y = p.start.year
    if pd.isna(p.ret) or p.ret.year != y:
        c["no_activation_in_the_same_season"] += 1; continue
    if (p.batter, y) not in G:
        c["no_games_that_season"] += 1; continue
    base, _ = baseline(p.batter, y, p.start - pd.Timedelta(days=30))
    if base is None:
        c["no_baseline"] += 1; continue
    later = P[(P["batter"] == p.batter) & (P["start"] > p.start)]["start"]
    stop = later.min() if len(later) else pd.Timestamp(f"{y}-12-31")
    if post_rows(p.batter, y, p.start, p.ret, stop) is None:
        c["no_game_after_return_before_a_later_placement"] += 1; continue
    c["stints"] += 1
post.update(c)
post["with_activation_in_the_same_season"] = post["placements"] - post["no_activation_in_the_same_season"]
assert post["stints"] == G_json["F4b"]["n_stints"], post
n110 = G_json["F4b"]["effects"]["all"]["bat_speed"]["1-10"]["n"]
post["with_competitive_swings_in_games_1_10"] = int(n110)

# ------------------------------------------------------------------ IL check: why placements were missing from the record-date file
isrc = open(os.path.join(HERE, "ajsm_il_coding.py"), encoding="utf-8").read()
isrc = isrc[:isrc.index('res = {"raw_lines"')]
scratch = os.path.join(RUN, "il_check")
os.makedirs(scratch, exist_ok=True)
sys.argv = ["ajsm_il_coding.py", RAW, ILO, os.path.join(scratch, "coded.csv"), os.path.join(scratch, "unused.json")]
NI = {"__name__": "__main__"}
exec(compile(isrc, "ajsm_il_coding.py (to the check)", "exec"), NI)
mf, pl = NI["missing_from_file"], NI["pl"]
rec = pd.to_datetime(mf["date"])
gap = (mf["day"] - rec).dt.days
il = {"missing_from_file": int(len(mf)),
      "missing_record_date_more_than_2_days_earlier": int((mf["date_fix"] == "record date earlier than effective date (effective date used)").sum()),
      "missing_year_typo_in_retroactive_date": int((mf["date_fix"] == "year typo in stated retroactive date (corrected)").sum()),
      "missing_record_date_1_or_2_days_earlier": int(((mf["date_fix"] == "none") & gap.between(1, 2)).sum()),
      "missing_other": int(((mf["date_fix"] == "none") & ~gap.between(1, 2)).sum()),
      "record_date_fixes_by_effective_year": pl.loc[pl["date_fix"] != "none", "day"].dt.year.value_counts().sort_index().to_dict(),
      "record_date_fixes_with_record_date_in_2023": int((pd.to_datetime(pl.loc[pl["date_fix"] != "none", "date"]).dt.year == 2023).sum())}
assert il["missing_from_file"] == json.load(open(os.path.join(HERE, "results", "ajsm_il_coding.json")))["api_placements_2024_2026_missing_from_file"]
assert sum(il[k] for k in ("missing_record_date_more_than_2_days_earlier", "missing_year_typo_in_retroactive_date",
                           "missing_record_date_1_or_2_days_earlier", "missing_other")) == il["missing_from_file"]
# added after the re-check of 2026-10-19: how the matched and missing placements make up the corrected file (Figure 1, 739)
mt, pl24 = NI["matched"], NI["pl24"]
il["file_matched_rows"] = int(len(mt))
il["file_matched_distinct_placements"] = int(mt[["batter", "day"]].drop_duplicates().shape[0])
il["file_matched_same_day_duplicates"] = il["file_matched_rows"] - il["file_matched_distinct_placements"]
il["mlb_placements_2024_2026"] = int(len(pl24))
_cf = json.load(open(os.path.join(HERE, "results", "ajsm_il_coding.json")))["corrected_il_file"]["placements"]
assert il["file_matched_distinct_placements"] + il["missing_from_file"] == il["mlb_placements_2024_2026"] == _cf, (il, _cf)
il["corrected_file_placements"] = int(_cf)
# the MLB records with a record date more than 2 days early that were not missing from the file (S4 note)
_early = pl[pl["date_fix"] == "record date earlier than effective date (effective date used)"]
_miss, _mtk = set(zip(mf["batter"], mf["day"])), set(zip(mt["batter"], mt["day"]))
_ufk = set(zip(NI["unmatched_file"]["batter"], NI["unmatched_file"]["day"]))
_nm = _early[[(b, d) not in _miss for b, d in zip(_early["batter"], _early["day"])]]
il["early_record_records"] = int(len(_early))
il["early_record_not_missing"] = int(len(_nm))
il["early_record_not_missing_in_file_on_effective_date"] = int(sum((b, d) in _mtk for b, d in zip(_nm["batter"], _nm["day"])))
il["early_record_not_missing_also_in_file_on_record_date"] = int(sum((b, pd.Timestamp(r)) in _ufk for b, r in zip(_nm["batter"], _nm["date"])))
assert il["early_record_records"] == il["missing_record_date_more_than_2_days_earlier"] + il["early_record_not_missing"], il
# added after the second re-check (2026-10-20): what the same-day duplicates among the matched rows are (S4)
_dk = mt[mt.duplicated(["batter", "day"], keep=False)][["batter", "day"]].drop_duplicates()
_rw = NI["raw"]
_rw = _rw[_rw["mlb"] & (_rw["kind"] == "placed")].assign(_rec=lambda x: pd.to_datetime(x["date"]), _eff=lambda x: pd.to_datetime(x["eff"]))
_double = _later = 0
for _b, _d in _dk.itertuples(index=False):
    _on = _rw[(_rw["batter"] == _b) & (_rw["_rec"] == _d)]
    if int((_on["_eff"] == _d).sum()) >= 2:
        _double += 1                                   # MLB lists the same placement twice on that date
    elif int((_on["_eff"] == _d).sum()) == 1 and bool((_on["_eff"] > _d).any()):
        _later += 1                                    # a later placement dated by its record date, the date of another placement
il["file_matched_dup_listed_twice_by_mlb"] = int(_double)
il["file_matched_dup_later_placement_on_its_record_date"] = int(_later)
assert _double + _later == il["file_matched_same_day_duplicates"], il
il["file_misdated_or_duplicated_total"] = int(len(NI["unmatched_file"])) + il["file_matched_same_day_duplicates"]
res = {"note": __doc__.split("\n\n")[0], "before_placement": pre, "after_return": post, "il_check": il}
json.dump(res, open(OUT, "w"), indent=1, default=int)
print(json.dumps(res, indent=1, default=int))
