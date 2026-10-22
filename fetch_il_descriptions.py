"""Pull the injured-list transaction text for every hitter in the public IL file from the MLB Stats API (plan addendum G).

python3 fetch_il_descriptions.py IL_FILE OUT_TXT

Uses /api/v1/people?personIds=...&hydrate=transactions(startDate=2023-01-01,endDate=2026-10-05), 20 hitters per call,
and keeps every transaction whose text contains "injured list" (all levels; ajsm_il_coding.py keeps the MLB ones).
Output lines: id|fullName|typeCode|date|effectiveDate|description. The file used for the AJSM analyses was pulled on
2026-10-05 with this request (through the MLB Stats API tool on the build machine) and has 2,067 lines.
"""
import json
import sys
import urllib.parse
import urllib.request

IL, OUT = sys.argv[1:3]
ids = [l.split(":")[0] for l in open(IL) if l.strip()]
API = "https://statsapi.mlb.com/api/v1/people"
lines = []
for k in range(0, len(ids), 20):
    q = urllib.parse.urlencode({"personIds": ",".join(ids[k:k + 20]),
                                "hydrate": "transactions(startDate=2023-01-01,endDate=2026-10-05)",
                                "fields": "people,id,fullName,transactions,date,effectiveDate,description,typeCode"})
    with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
        js = json.load(r)
    for p in js.get("people", []):
        for t in p.get("transactions", []):
            d = t.get("description", "")
            if "injured list" in d.lower():
                lines.append("|".join([str(p["id"]), p.get("fullName", ""), t.get("typeCode", ""), t.get("date", ""),
                                       t.get("effectiveDate", ""), d.replace("|", "/")]))
open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(len(ids), "hitters,", len(lines), "injured-list transactions")
