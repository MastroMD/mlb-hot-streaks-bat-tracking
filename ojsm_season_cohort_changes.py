"""Which season-level cohorts hold the hitter-seasons whose injury families changed when IL records were merged into stints
(post hoc, third independent check of 2026-10-20).

python3 ojsm_season_cohort_changes.py IL_STINTS FG_BATTRACK_CSV FG_POSITION_XLSX FG_2026_CSV PUBLISHED_SEASON_JSON COMPARE_JSON OUT_JSON

Runs ajsm_season.py's own code on IL_STINTS (the record-date stint file of the prespecified season-level analyses), stops
unless its output equals PUBLISHED_SEASON_JSON (repo/results/stints_2026-10-20/ajsm_season.json), and reports, for each
hitter-season listed under "family changes, record-date file" in COMPARE_JSON (ojsm_stints_compare.json), whether it is in
the F1 primary cohort, the F1 sensitivity cohort that adds 2024, and the F2 cohort, with its F2 exposure (first family) and
any-hand/wrist indicator.
"""
import json
import os
import sys

IL, BT, XL, F26, PUB, CMP, OUT = sys.argv[1:8]
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ajsm_season.py")
tmp = os.path.join(os.path.dirname(os.path.abspath(OUT)), "ajsm_season_rerun.json")
sys.argv = [SRC, IL, BT, XL, F26, tmp]
ns = {"__name__": "__main__", "__file__": SRC}
exec(compile(open(SRC, encoding="utf-8").read(), SRC, "exec"), ns)
a, b = json.load(open(tmp)), json.load(open(PUB))
a["inputs"].pop("il", None); il_pub = b["inputs"].pop("il", None)
assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), "rerun differs from the published ajsm_season.json"
prim, sa, D = ns["prim"], ns["sa"], ns["D"]
rows = []
for c in json.load(open(CMP))["family changes, record-date file"]:
    bb, y = c["batter"], c["season"]
    d = D[(D["batter"] == bb) & (D["Y"] == y)]
    rows.append({"batter": bb, "season": y, "records": c["records"], "stints": c["stints"],
                 "in_F1_primary": bool(((prim["batter"] == bb) & (prim["Y"] == y)).any()),
                 "in_F1_sensitivity_add_2024": bool(((sa["batter"] == bb) & (sa["Y"] == y)).any()),
                 "in_F2": bool(len(d)), "F2_first_family": str(d["fam"].iloc[0]) if len(d) else None,
                 "F2_any_hand_wrist": int(d["any_hw"].iloc[0]) if len(d) else None})
json.dump({"note": " ".join(__doc__.split("\n\n")[1].split()), "il_sha16": il_pub, "rerun_equals_published": True, "hitter_seasons": rows},
          open(OUT, "w"), indent=1)
print(json.dumps(rows, indent=1))
