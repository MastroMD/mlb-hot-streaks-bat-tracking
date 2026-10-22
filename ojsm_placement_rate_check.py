"""Is the lower IL placement rate of 2024 in MLB's own records? (author items of 2026-10-21, PLAN_ADDENDUM_2026-10-17h.md
item 14; post hoc, descriptive)

python3 ojsm_placement_rate_check.py TRANSACTIONS_JSON CORRECTED_IL_FILE OUT_JSON

TRANSACTIONS_JSON  transactions_2024-01-01_2026-10-05.json (MLB Stats API date-range pull of every transaction; repo/data_public/statsapi/)
CORRECTED_IL_FILE  repo/data_public/il_transactions_hitters_2024_2026_corrected.txt (placement records, from the players'
                   transaction histories, a separate MLB Stats API route; ajsm_il_coding.py)

Counts, by season (March through September, effective date), the injured-list placements by Major League clubs in the
date-range pull, for every position player and every pitcher (the position abbreviation of the transaction text), one per
player and date, and those of the hitters covered by the IL file, against the placement records of the corrected file.
"""
import json
import re
import sys

import pandas as pd

TRX, ILF, OUT = sys.argv[1:4]
MLB = ["Arizona Diamondbacks", "Athletics", "Oakland Athletics", "Atlanta Braves", "Baltimore Orioles", "Boston Red Sox",
       "Chicago Cubs", "Chicago White Sox", "Cincinnati Reds", "Cleveland Guardians", "Colorado Rockies", "Detroit Tigers",
       "Houston Astros", "Kansas City Royals", "Los Angeles Angels", "Los Angeles Dodgers", "Miami Marlins",
       "Milwaukee Brewers", "Minnesota Twins", "New York Mets", "New York Yankees", "Philadelphia Phillies",
       "Pittsburgh Pirates", "San Diego Padres", "San Francisco Giants", "Seattle Mariners", "St. Louis Cardinals",
       "Tampa Bay Rays", "Texas Rangers", "Toronto Blue Jays", "Washington Nationals"]   # as in ajsm_il_coding.py
PITCH = {"RHP", "LHP", "P"}
rows = []
for t in json.load(open(TRX)):
    m = re.match(r"(.+?) placed (\S+) (.+?) on the (\d+)-day injured list", t.get("description") or "")
    if not m or m.group(1) not in MLB or not (t.get("person") or {}).get("id"):
        continue
    rows.append({"id": int(t["person"]["id"]), "pos": m.group(2), "days": int(m.group(4)),
                 "eff": pd.Timestamp((t.get("effectiveDate") or t.get("date"))[:10]), "tid": t.get("id")})
R = pd.DataFrame(rows).drop_duplicates("tid")
R = R[(R["eff"].dt.month >= 3) & (R["eff"].dt.month <= 9)]
R["season"] = R["eff"].dt.year
R["pitcher"] = R["pos"].isin(PITCH)
R = R.drop_duplicates(["id", "eff"])
covered = set(int(l.split(":")[0]) for l in open(ILF) if l.strip())
fileP = {}
for l in open(ILF):
    if l.strip():
        for e in l.strip().split(":")[1].split():
            if e[6] == "P":
                fileP[2000 + int(e[:2])] = fileP.get(2000 + int(e[:2]), 0) + 1
by = lambda d: {int(k): int(v) for k, v in d.groupby("season").size().items()}
out = {"note": " ".join(__doc__.split("\n\n")[2].split()),
       "position_players": by(R[~R["pitcher"]]), "pitchers": by(R[R["pitcher"]]),
       "position_players_covered_by_the_il_file": by(R[~R["pitcher"] & R["id"].isin(covered)]),
       "corrected_il_file_placement_records": {int(k): v for k, v in sorted(fileP.items())},
       "position_players_by_list": {str(d): by(R[~R["pitcher"] & (R["days"] == d)]) for d in sorted(R["days"].unique())}}
for k in ("position_players", "pitchers", "position_players_covered_by_the_il_file", "corrected_il_file_placement_records"):
    v = out[k]
    out[k + "_2024_over_mean_2025_2026"] = v[2024] / ((v[2025] + v[2026]) / 2)
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1))
