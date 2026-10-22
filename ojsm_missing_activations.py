"""IL placements followed by Major League games with no activation in the IL file (post hoc, third independent check of
2026-10-20), and a copy of the IL file with those activations added, for a sensitivity analysis.

python3 ojsm_missing_activations.py IL_FILE STATCAST_DIR TRANSACTIONS_JSON OUT_IL OUT_JSON

IL_FILE            repo/data_public/il_transactions_hitters_2024_2026_corrected_stints.txt
STATCAST_DIR       the Statcast season files (regular-season games played, by hitter)
TRANSACTIONS_JSON  transactions_2024-01-01_2026-10-05.json (MLB Stats API, every transaction; repo/data_public/statsapi/)

The IL file takes activations from MLB transactions whose text names the injured list (fetch_il_descriptions.py,
ajsm_il_coding.py). A placement counts as followed by an activation when one is listed after it and before the hitter's
next placement (as in ajsm_season.py and ajsm_game_ojsm.py). Here a placement with no such activation, after which the
hitter played a regular-season game in the same season before his next placement, gets an activation dated by the first
of those games. Each is classified by the MLB transaction that returned the hitter (the last one on or before that game
that activates, reinstates or recalls him): an activation from the All-Star reserve list, an activation that names no
list, a recall from a minor-league club, or no transaction. OUT_IL is IL_FILE with these activations added; IL_FILE is
not changed.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ILF, SC, TRX, OUT_IL, OUT_J = sys.argv[1:6]
FAM = {"O": "oblique", "H": "hamstring", "W": "hand_wrist", "X": "other"}
files = [f for f in sorted(glob.glob(os.path.join(SC, "*.parquet"))) if any(y in os.path.basename(f) for y in ("2024", "2025", "2026"))]
g = pd.concat([pd.read_parquet(f, columns=["game_date", "game_type", "batter", "game_pk"]) for f in files], ignore_index=True)
g = g[g["game_type"] == "R"].drop_duplicates(["batter", "game_pk"])
g["game_date"] = pd.to_datetime(g["game_date"])
g["batter"] = g["batter"].astype(int)
GD = {b: np.sort(x["game_date"].to_numpy()) for b, x in g.groupby("batter")}
TX = {}
for t in json.load(open(TRX)):
    pid = (t.get("person") or {}).get("id")
    if pid is not None:
        TX.setdefault(int(pid), []).append(((t.get("effectiveDate") or t.get("date") or "")[:10], t.get("description") or ""))


def category(b, s, first):
    c = [(d, x) for d, x in TX.get(b, []) if d and str(s.date()) <= d <= str(first.date())
         and re.search(r"\b(activated|reinstated|recalled)\b", x)]
    if not c:
        return "no transaction recorded", ""
    d, x = sorted(c)[-1]
    if "reserve list" in x:
        return "activated from the All-Star reserve list", f"{d} {x}"
    if "recalled" in x:
        return "recalled from a minor-league club", f"{d} {x}"
    if "injured list" not in x:
        return "activated without naming a list", f"{d} {x}"
    return "activation naming the injured list, not by a Major League club", f"{d} {x}"


lines, rows = [], []
for line in open(ILF):
    line = line.strip()
    if not line:
        continue
    b, ev = line.split(":")
    bi, evs = int(b), ev.split()
    P = [(pd.Timestamp("20" + e[:6]), e) for e in evs if e[6] == "P"]
    A = [pd.Timestamp("20" + e[:6]) for e in evs if e[6] == "A"]
    add = []
    for k, (s, e) in enumerate(P):
        nxt = P[k + 1][0] if k + 1 < len(P) else pd.Timestamp("2100-01-01")
        if any(s <= a <= nxt for a in A):
            continue
        t = GD.get(bi, np.array([], dtype="datetime64[ns]"))
        end = min(nxt, pd.Timestamp(f"{s.year}-12-31"))
        played = t[(t > np.datetime64(s)) & (t < np.datetime64(end))]
        if not len(played):
            continue
        first = pd.Timestamp(played[0])
        cat, txt = category(bi, s, first)
        add.append(f"{first:%y%m%d}A")
        rows.append({"batter": bi, "placement": e, "family": FAM[e[7]], "first_game_after": str(first.date()),
                     "games_before_next_placement": int(len(played)), "category": cat, "transaction": txt})
    lines.append(f"{b}:{' '.join(sorted(set(evs + add)))}")
open(OUT_IL, "w").write("\n".join(lines) + "\n")
R = pd.DataFrame(rows)
out = {"note": __doc__.split("\n\n")[1].replace("\n", " "), "il_file": ILF, "out_il": OUT_IL,
       "placements_with_games_but_no_activation": int(len(R)),
       "by_family": R["family"].value_counts().to_dict(), "by_category": R["category"].value_counts().to_dict(),
       "records": rows}
json.dump(out, open(OUT_J, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("placements_with_games_but_no_activation", "by_family", "by_category")}, indent=1))
