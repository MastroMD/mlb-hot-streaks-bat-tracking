"""Pull (1) all MLB Stats API transactions 2024-01-01..2026-10-05 (monthly chunks) and (2) regular-season schedules
2024-2026 with dayNight/doubleheader fields. Public data. Output in /home/claude/work/statsapi/."""
import json, time, urllib.request
from datetime import date, timedelta
OUT = "/home/claude/work/statsapi"
def get(u):
    for i in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=120) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            last = e; time.sleep(5 * (i + 1))
    raise last
tx = []
cur = date(2024, 1, 1)
while cur <= date(2026, 10, 5):
    nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
    end = min(nxt - timedelta(days=1), date(2026, 10, 5))
    j = get(f"https://statsapi.mlb.com/api/v1/transactions?startDate={cur}&endDate={end}")
    tx += j.get("transactions", [])
    print(cur, len(j.get("transactions", [])), flush=True)
    cur = nxt
json.dump(tx, open(f"{OUT}/transactions_2024-01-01_2026-10-05.json", "w"))
sched = []
for yr in (2024, 2025, 2026):
    j = get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={yr}&gameType=R")
    for d in j["dates"]:
        for g in d["games"]:
            sched.append({k: g.get(k) for k in ["gamePk", "gameDate", "officialDate", "dayNight", "doubleHeader", "gameNumber", "seriesGameNumber"]}
                         | {"home": g["teams"]["home"]["team"]["id"], "away": g["teams"]["away"]["team"]["id"], "status": g["status"].get("detailedState")})
    print(yr, len(sched), flush=True)
json.dump(sched, open(f"{OUT}/schedule_2024_2026.json", "w"))
print("DONE", len(tx), len(sched))
