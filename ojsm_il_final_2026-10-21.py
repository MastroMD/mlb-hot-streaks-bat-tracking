"""Final IL inputs (author items of 2026-10-21, PLAN_ADDENDUM_2026-10-17h.md item 14; post hoc).

python3 ojsm_il_final_2026-10-21.py STINTS_JSON CORR_STINTS ORIG_STINTS CODED_CSV OUT_CORR_STINTS OUT_ORIG_STINTS OUT_CODED_CSV OUT_JSON

STINTS_JSON  repo/results/stints_2026-10-20/ojsm_il_stints.json (ojsm_il_stints.py: the records merged into each stint)
CORR_STINTS  repo/data_public/il_transactions_hitters_2024_2026_corrected_stints.txt
ORIG_STINTS  repo/data_public/il_transactions_hitters_2024_2026_stints.txt
CODED_CSV    repo/data_public/il_placements_coded_2023_2026.csv

(1) A stint into which a record naming a different injury was merged (in both files, 1 stint: a hamate fracture placed on
    the 60-day list during a stint for a foot contusion) is classified by that later injury, the one for which the hitter
    stayed on the IL until his activation: the family letter of the stint in both stint files, and the injury text, region,
    mechanism, impact flag and text family of its first record in the coded file, are those of the later record. A merged
    record with the same injury text (or none) changes nothing.
(2) Body region: the region rule of ajsm_il_coding.py matches 'side' (oblique/abdominal) before 'rib' (chest or rib), so
    the 2 placements whose text names a rib fracture 'on the left side' or on the 'right side' were coded oblique/abdominal;
    they are coded chest or rib. No other placement text contains 'side'.
The input files are not changed. Activations missing from the IL file are added afterwards by ojsm_missing_activations.py.
"""
import json
import re
import sys

import pandas as pd

SJ, CORR, ORIG, CODED, OUT_CORR, OUT_ORIG, OUT_CODED, OUT_J = sys.argv[1:9]
ST = json.load(open(SJ))
out = {"note": " ".join(__doc__.split("\n\n", 1)[1].split()), "relabeled_stints": {}, "rib_recoded": []}

# (1) stints classified by a later record that names a different injury
relabel = {}
for key in ("original", "corrected"):
    rel = []
    for m in ST[key]["merged_records"]:
        a, b = (m["stint_text"] or "").strip().lower(), (m["merged_text"] or "").strip().lower()
        if a and b and a != b:
            rel.append({"batter": m["batter"], "stint_code": m["stint_code"], "stint_text": m["stint_text"],
                        "merged_code": m["merged_code"], "merged_text": m["merged_text"], "merged_date": m["merged_date"],
                        "new_code": m["stint_code"][:7] + m["merged_code"][7] + m["stint_code"][8:]})
    out["relabeled_stints"][key] = rel
    relabel[key] = {(r["batter"], r["stint_code"]): r["new_code"] for r in rel}
assert [(r["batter"], r["stint_code"], r["new_code"]) for r in out["relabeled_stints"]["corrected"]] == [(682657, "240325PX10", "240325PW10")]
assert out["relabeled_stints"]["original"] == out["relabeled_stints"]["corrected"]


def rewrite(src, dst, rel):
    lines, n = [], 0
    for line in open(src):
        line = line.strip()
        if not line:
            continue
        b, ev = line.split(":")
        evs = ev.split()
        new = [rel.get((int(b), e), e) for e in evs]
        n += sum(x != y for x, y in zip(evs, new))
        lines.append(f"{b}:{' '.join(new)}")
    open(dst, "w").write("\n".join(lines) + "\n")
    return n


out["stint_codes_changed"] = {"corrected": rewrite(CORR, OUT_CORR, relabel["corrected"]), "original": rewrite(ORIG, OUT_ORIG, relabel["original"])}
assert out["stint_codes_changed"] == {"corrected": 1, "original": 1}

# coded file: the stint's first record takes the coding of the later record; rib fractures coded by 'side'
cd = pd.read_csv(CODED, dtype={"il_days": str})
for r in out["relabeled_stints"]["corrected"]:
    first = cd.index[(cd["batter"] == r["batter"]) & (cd["day"] == "20" + r["stint_code"][:2] + "-" + r["stint_code"][2:4] + "-" + r["stint_code"][4:6])]
    later = cd.index[(cd["batter"] == r["batter"]) & (cd["day"] == r["merged_date"])]
    assert len(first) == 1 and len(later) == 1
    r["coded_before"] = cd.loc[first[0], ["text", "region", "mechanism", "impact", "fam4_from_text"]].astype(str).to_dict()
    for col in ("text", "region", "mechanism", "impact", "fam4_from_text"):
        cd.at[first[0], col] = cd.at[later[0], col]
    r["coded_after"] = cd.loc[first[0], ["text", "region", "mechanism", "impact", "fam4_from_text"]].astype(str).to_dict()
t = cd["text"].fillna("").str.lower()
side = t.str.contains(r"\bside\b")
rib = side & t.str.contains(r"\brib") & (cd["region"] == "oblique_abdominal")
assert int(side.sum()) == int(rib.sum()) == 2
for i in cd.index[rib]:
    out["rib_recoded"].append({"batter": int(cd.at[i, "batter"]), "name": cd.at[i, "name"], "day": cd.at[i, "day"], "text": cd.at[i, "text"],
                               "region_before": cd.at[i, "region"], "region_after": "chest_rib"})
    cd.at[i, "region"] = "chest_rib"
cd.to_csv(OUT_CODED, index=False)
json.dump(out, open(OUT_J, "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
