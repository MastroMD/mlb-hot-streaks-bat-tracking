"""Code injured-list placements by body region and mechanism from the MLB Stats API transaction text, and check the
public IL file used by the paper against it (plan addendum G).

python3 ajsm_il_coding.py RAW_DESCRIPTIONS IL_FILE OUT_CSV OUT_JSON

RAW_DESCRIPTIONS  id|fullName|typeCode|date|effectiveDate|description, one injured-list transaction per line, from
                  fetch_il_descriptions.py (people endpoint, hydrate=transactions, 2023-01-01 to 2026-10-05)
IL_FILE           repo/data_public/il_transactions_hitters_2024_2026.txt (the paper's coding: families O, H, W, X)

Only Major League placements are coded (the description starts with an MLB club). When a placement has no injury text,
the injury text of the next transfer of the same stint is used.
"""
import json
import re
import sys

import pandas as pd

RAW, ILF, OUT_CSV, OUT_JSON = sys.argv[1:5]
OUT_IL = sys.argv[5] if len(sys.argv) > 5 else None     # corrected IL file in the paper's format
MLB = ["Arizona Diamondbacks", "Athletics", "Oakland Athletics", "Atlanta Braves", "Baltimore Orioles", "Boston Red Sox",
       "Chicago Cubs", "Chicago White Sox", "Cincinnati Reds", "Cleveland Guardians", "Colorado Rockies", "Detroit Tigers",
       "Houston Astros", "Kansas City Royals", "Los Angeles Angels", "Los Angeles Dodgers", "Miami Marlins",
       "Milwaukee Brewers", "Minnesota Twins", "New York Mets", "New York Yankees", "Philadelphia Phillies",
       "Pittsburgh Pirates", "San Diego Padres", "San Francisco Giants", "Seattle Mariners", "St. Louis Cardinals",
       "Tampa Bay Rays", "Texas Rangers", "Toronto Blue Jays", "Washington Nationals"]
REGION = [  # first match wins
    ("head_face", r"concussion|convussion|conussion|\bhead\b|face|facial|jaw|\beye\b|nasal|\bnose\b|orbital"),
    ("hand_wrist", r"\bhand\b|wrist|thumb|finger|hamm?ate|metacarpal|knuckle|scaphoid|\bpalm\b"),
    ("oblique_abdominal", r"ob?l?ique|intercostal|abdominal|abdomen|\bside\b"),
    ("hamstring", r"hamstring"),
    ("elbow_forearm_arm", r"elbow|forearm|ulnar|epicondyl|tricep|bicep|\barm\b|(?<!hip )flexor|tommy john"),
    ("shoulder", r"shoulder|rotator|labrum|clavicle|scapula|\bac joint|\blat\b|latissimus"),
    ("back_neck", r"\bback\b|spine|spinal|lumbar|thoracic|\bneck\b|cervical|\bdisc\b|sacroiliac|\bsi joint"),
    ("chest_rib", r"\brib|chest|pectoral|sternum|costochondral|\bsc joint"),
    ("hip_groin", r"groin|adductor|abductor|\bhip\b|hip flexor|pelvi|hernia|\bcore\b|glute"),
    ("thigh_quad", r"\bquad|thigh|femoral|femur"),
    ("knee", r"knee|patell|meniscus|\bacl\b|\bmcl\b"),
    ("leg_ankle_foot", r"calf|calif|haglund|achilles|ankle|\bfoot\b|\btoe|heel|plantar|\bshin|tibia|fibula|\bleg\b"),
    ("illness", r"illness|influenza|virus|viral|covid|infection|appendic|kidney|gastro|flu\b|sinus|pneumonia"),
]
MECH = [
    ("surgery", r"surgery|reconstruction|excision|revision"),
    ("fracture", r"fractur|tracture|broken|break\b"),
    ("impact", r"contusion|bruise|laceration|concussion|convussion|conussion|dislocat|hit by|bone bruise"),
    ("ligament_joint", r"sprain|ligament|\bucl\b|\bacl\b|\bmcl\b|meniscus|labr|cartilage|subluxation|torn (?!.*(muscle|strain))"),
    ("muscle_strain", r"strain|\btear\b|torn|pull|hernia|core muscle"),
    ("inflammatory_overuse", r"inflammation|tendinitis|tendonitis|tendinopathy|tendon|soreness|\bsore\b|discomfort|tightness|"
                             r"impingement|epicondylitis|spasm|bursitis|irritation|stress reaction|fasciitis|neuritis|nerve|"
                             r"herniat|\bdisc\b|stenosis|spondyl|synovitis|cyst|bone spur|loose bod|pain|tenderness|headache|inflamm|fascitis"),
    ("illness", r"illness|influenza|virus|viral|covid|infection|appendic|kidney|gastro|flu\b|sinus|pneumonia"),
]
FAM4 = {"hand_wrist": "W", "oblique_abdominal": None, "hamstring": "H"}


def injury_text(desc):
    m = re.search(r"injured list(?: retroactive to [A-Z][a-z]+ \d+, \d{4})?\.\s*(.*)$", desc)
    return (m.group(1).strip() if m else "").rstrip(".")


def code(text, table):
    low = text.lower()
    for name, pat in table:
        if re.search(pat, low):
            return name
    return "unspecified" if not low else "other"


raw = pd.DataFrame([l.rstrip("\n").split("|") for l in open(RAW, encoding="utf-8")],
                   columns=["batter", "name", "type", "date", "eff", "desc"])
raw["batter"] = raw["batter"].astype(int)
raw = raw.drop_duplicates(["batter", "date", "eff", "desc"])
raw["mlb"] = raw["desc"].apply(lambda s: any(s.startswith(t + " ") for t in MLB))
raw["kind"] = raw["desc"].str.extract(r"\s(placed|activated|transferred|reinstated)\s", expand=False)
raw["day"] = pd.to_datetime(raw["eff"].where(raw["eff"] != "", raw["date"]))
_rec = pd.to_datetime(raw["date"])
raw["date_fix"] = "none"
raw.loc[(raw["day"] - _rec).dt.days > 2, "date_fix"] = "record date earlier than effective date (effective date used)"
_typo = (_rec - raw["day"]).dt.days > 300
raw.loc[_typo, "day"] = raw.loc[_typo, "day"] + pd.DateOffset(years=1)
raw.loc[_typo, "date_fix"] = "year typo in stated retroactive date (corrected)"
raw["text"] = raw["desc"].apply(injury_text)
raw = raw.sort_values(["batter", "day"]).reset_index(drop=True)
pl = raw[raw["mlb"] & (raw["kind"] == "placed")].copy()
tr = raw[raw["mlb"] & (raw["kind"] == "transferred")]
ac = raw[raw["mlb"] & (raw["kind"].isin(["activated", "reinstated"]))]
filled = 0
for i, r in pl.iterrows():
    if r["text"]:
        continue
    nxt_pl = pl[(pl["batter"] == r["batter"]) & (pl["day"] > r["day"])]["day"].min()
    t = tr[(tr["batter"] == r["batter"]) & (tr["day"] >= r["day"]) & ((tr["day"] < nxt_pl) if pd.notna(nxt_pl) else True)]
    t = t[t["text"] != ""]
    if len(t):
        pl.at[i, "text"] = t["text"].iloc[0]; filled += 1
pl["region"] = pl["text"].apply(lambda s: code(s, REGION))
pl["mechanism"] = pl["text"].apply(lambda s: code(s, MECH))
pl["impact"] = pl["mechanism"].isin(["fracture", "impact"]) | (pl["region"] == "head_face")
pl["fam4_from_text"] = pl["region"].map({"hand_wrist": "W", "hamstring": "H"}).fillna("X")
pl.loc[pl["text"].str.lower().str.contains("ob?l?ique|intercostal"), "fam4_from_text"] = "O"
pl["il_days"] = pl["desc"].str.extract(r"(\d+)-day", expand=False)
pl = pl.drop_duplicates(["batter", "day"])
out = pl[["batter", "name", "day", "il_days", "text", "region", "mechanism", "impact", "fam4_from_text", "desc"]].copy()
out["day"] = out["day"].dt.strftime("%Y-%m-%d")
out.to_csv(OUT_CSV, index=False)

# ------------------------------------------------------------------ check the paper's IL file
F = []
for line in open(ILF):
    b, ev = line.strip().split(":")
    for e in ev.split():
        if e[6] == "P":
            F.append({"batter": int(b), "day": pd.Timestamp("20" + e[:6]), "fam_file": e[7]})
F = pd.DataFrame(F)
pl24 = pl[pl["day"].dt.year >= 2024]
m = F.merge(pl24[["batter", "day", "fam4_from_text", "text", "region", "mechanism", "desc"]], on=["batter", "day"], how="left", indicator=True)
unmatched_file = m[m["_merge"] == "left_only"]
near = []
for r in unmatched_file.itertuples():           # a placement within 3 days (retroactive dating differences)
    c = pl24[(pl24["batter"] == r.batter) & ((pl24["day"] - r.day).abs() <= pd.Timedelta(days=3))]
    near.append(len(c) > 0)
matched = m[m["_merge"] == "both"]
dis = matched[matched["fam_file"] != matched["fam4_from_text"]]
missing_from_file = pl24.merge(F, on=["batter", "day"], how="left", indicator=True)
missing_from_file = missing_from_file[(missing_from_file["_merge"] == "left_only") & (missing_from_file["day"] <= "2026-09-30")]
res = {"raw_lines": int(len(raw)), "mlb_placements_2023_2026": int(len(pl)),
       "mlb_placements_by_year": pl["day"].dt.year.value_counts().sort_index().to_dict(),
       "placements_text_filled_from_transfer": filled,
       "placements_without_injury_text": int((pl["text"] == "").sum()),
       "file_placements": int(len(F)), "file_matched_same_day": int(len(matched)),
       "file_unmatched": int(len(unmatched_file)), "file_unmatched_with_placement_within_3_days": int(sum(near)),
       "family_disagreements": int(len(dis)),
       "family_disagreement_examples": dis[["batter", "day", "fam_file", "fam4_from_text", "text"]].astype(str).head(40).to_dict(orient="records"),
       "family_confusion": pd.crosstab(matched["fam_file"], matched["fam4_from_text"]).to_dict(),
       "api_placements_2024_2026_missing_from_file": int(len(missing_from_file)),
       "missing_examples": missing_from_file[["batter", "day", "text"]].astype(str).head(30).to_dict(orient="records"),
       "unmatched_file_examples": unmatched_file[["batter", "day", "fam_file"]].astype(str).head(30).to_dict(orient="records"),
       "region_counts_2024_2026": pl24["region"].value_counts().to_dict(),
       "mechanism_counts_2024_2026": pl24["mechanism"].value_counts().to_dict(),
       "impact_share_2024_2026": float(pl24["impact"].mean())}
res["date_fixes_mlb_placements"] = pl["date_fix"].value_counts().to_dict() if "date_fix" in pl else {}
if OUT_IL:
    ids = [int(l.split(":")[0]) for l in open(ILF)]
    fam = lambda d_: ("O" if re.search(r"ob?l?ique|intercostal", d_.lower()) else "H" if "hamstring" in d_.lower() else
                      "W" if re.search(r"hand|wrist|thumb|finger|hamm?ate|metacarpal", d_.lower()) else "X")
    lo, hi = pd.Timestamp("2024-02-01"), pd.Timestamp("2026-09-30")
    P_ = pl[(pl["day"] >= lo) & (pl["day"] <= hi) & pl["batter"].isin(ids)]
    A_ = ac[(ac["day"] >= lo) & (ac["day"] <= hi) & ac["batter"].isin(ids)].drop_duplicates(["batter", "day"])
    ev = {}
    for r in P_.itertuples():
        txt = r.desc.split("injured list", 1)[1] if "injured list" in r.desc else ""
        ev.setdefault(r.batter, []).append(f"{r.day:%y%m%d}P{fam(txt)}{r.il_days or ''}")
    for r in A_.itertuples():
        ev.setdefault(r.batter, []).append(f"{r.day:%y%m%d}A")
    open(OUT_IL, "w").write("\n".join(f"{k}:{' '.join(sorted(set(v)))}" for k, v in sorted(ev.items())) + "\n")
    res["corrected_il_file"] = {"path": OUT_IL, "placements": int(len(P_)), "activations": int(len(A_)), "hitters": len(ev)}
json.dump(res, open(OUT_JSON, "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if not k.endswith("examples")}, indent=1, default=str))
