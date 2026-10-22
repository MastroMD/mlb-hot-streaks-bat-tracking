"""Prior-season plate appearances of the season-level risk-factor cohort: median and interquartile range for Table 1
(author items of 2026-10-21, PLAN_ADDENDUM_2026-10-17h.md item 14; descriptive).

python3 ojsm_f1_pa_iqr.py IL_FILE FG_BATTRACK_CSV FG_POSITION_XLSX FG_2026_CSV PUBLISHED_SEASON_JSON OUT_JSON

Runs ajsm_season.py's own code on IL_FILE (the file of PUBLISHED_SEASON_JSON), stops unless its output equals
PUBLISHED_SEASON_JSON, and reports the quartiles of prior-season PA in the F1 primary cohort (ajsm_season.py stores only the
median).
"""
import json
import os
import sys

IL, BT, XL, F26, PUB, OUT = sys.argv[1:7]
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ajsm_season.py")
tmp = os.path.join(os.path.dirname(os.path.abspath(OUT)), "ajsm_season_rerun_pa_iqr.json")
sys.argv = [SRC, IL, BT, XL, F26, tmp]
ns = {"__name__": "__main__", "__file__": SRC}
exec(compile(open(SRC, encoding="utf-8").read(), SRC, "exec"), ns)
a, b = json.load(open(tmp)), json.load(open(PUB))
assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), "rerun differs from the published ajsm_season.json"
pa = ns["prim"]["pa_prev"]
assert float(pa.median()) == b["F1"]["cohort"]["pa_prev_median"] and len(pa) == b["F1"]["cohort"]["hitter_seasons"]
out = {"note": " ".join(__doc__.split("\n\n")[1].split()), "il_sha16": b["inputs"].get("il"), "n": int(len(pa)),
       "median": float(pa.median()), "q1": float(pa.quantile(0.25)), "q3": float(pa.quantile(0.75)), "min": float(pa.min()), "max": float(pa.max())}
json.dump(out, open(OUT, "w"), indent=1)
os.remove(tmp)
print(json.dumps(out, indent=1))
