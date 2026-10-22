"""Author decision of 2026-10-20 (PLAN_ADDENDUM_2026-10-17h.md, item 12): count IL stints, not IL placement records.

python3 ojsm_il_stints.py IL_ORIGINAL IL_CORRECTED CODED_CSV OUT_ORIGINAL OUT_CORRECTED OUT_JSON

MLB's public transaction records list some IL moves made during an ongoing IL stint as new placements: a placement on the
60-day list while the hitter is already on the IL (in effect a transfer), a near-duplicate record 1 or 2 days after the
placement, or the same injury placed again (for example by the new club after a trade). The analysis code treats every
placement record as a new placement, so such a record ends the previous placement's stint without an activation and starts a
new one. This script writes copies of both IL files (the record-date file of the prespecified season-level analyses and the
corrected file of the game-level analyses) in which each such record is merged into its stint:

  a placement record is dropped when the hitter's previous event in the file is a placement of the same season (no activation
  in between) and the record (a) is a 60-day placement, (b) follows the previous placement record within 2 days, or (c) names
  the same injury as the first record of the stint (transaction text of CODED_CSV, by hitter and date).

The stint keeps the date and the injury family of its first record and ends at the next activation. Other placements without
an activation since the previous placement are kept (they name a different injury, or an activation is missing from the
records) and are listed in OUT_JSON with the merged records. The input files are not changed.
"""
import json
import sys

import pandas as pd

ILO, ILC, CODED, OUTO, OUTC, OUTJ = sys.argv[1:7]
cd = pd.read_csv(CODED, parse_dates=["day"]).drop_duplicates(["batter", "day"])
TEXT = {(int(b), d.normalize()): (str(t).strip().lower() if isinstance(t, str) else "") for b, d, t in zip(cd["batter"], cd["day"], cd["text"])}
FAMNAME = {"O": "oblique", "H": "hamstring", "W": "hand_wrist", "X": "other"}


def merge(path, out_path):
    lines_out, merged, kept = [], [], []
    n_in = n_out = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        b, ev = line.split(":")
        bi = int(b)
        keep, in_stint, first, last_p = [], False, None, None
        for e in ev.split():
            d = pd.Timestamp("20" + e[:6])
            if e[6] != "P":
                keep.append(e)
                in_stint = False
                continue
            n_in += 1
            txt = TEXT.get((bi, d), "")
            if in_stint and first["date"].year == d.year:
                why = ("60-day placement" if e[8:] == "60" else
                       "within 2 days of the previous placement record" if (d - last_p).days <= 2 else
                       "same injury as the first record of the stint" if (txt and txt == first["text"]) else None)
                if why:
                    merged.append({"batter": bi, "stint_start": str(first["date"].date()), "stint_code": first["code"], "stint_text": first["text"],
                                   "merged_date": str(d.date()), "merged_code": e, "merged_text": txt, "reason": why,
                                   "family_differs": bool(e[7] != first["code"][7] and txt != "")})
                    last_p = d
                    continue
                kept.append({"batter": bi, "previous_stint_start": str(first["date"].date()), "previous_text": first["text"],
                             "kept_date": str(d.date()), "kept_code": e, "kept_text": txt})
            keep.append(e)
            n_out += 1
            in_stint, first, last_p = True, {"date": d, "code": e, "text": txt}, d
        lines_out.append(f"{b}:{' '.join(keep)}")
    open(out_path, "w").write("\n".join(lines_out) + "\n")
    by_reason = pd.Series([m["reason"] for m in merged]).value_counts().to_dict() if merged else {}
    return {"input": path, "output": out_path, "placement_records": n_in, "stints": n_out, "merged": len(merged),
            "merged_by_reason": {k: int(v) for k, v in by_reason.items()},
            "merged_naming_a_different_family": int(sum(m["family_differs"] for m in merged)),
            "kept_without_activation_same_season": len(kept), "merged_records": merged, "kept_records": kept}


res = {"note": __doc__.split("\n\n")[0], "original": merge(ILO, OUTO), "corrected": merge(ILC, OUTC)}
for k in ("original", "corrected"):
    r = res[k]
    assert r["placement_records"] - r["merged"] == r["stints"], r
json.dump(res, open(OUTJ, "w"), indent=1)
for k in ("original", "corrected"):
    r = res[k]
    print(k, {x: r[x] for x in ("placement_records", "stints", "merged", "merged_by_reason", "merged_naming_a_different_family", "kept_without_activation_same_season")})
