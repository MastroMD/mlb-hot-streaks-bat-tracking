"""Regenerate the headline table in README.md from results/paper_results.json (between the headline markers).
python3 render_readme.py"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, "results", "paper_results.json")))
T = P["paper"]["tables"]["T1"]
A, X = P["abstract"], P["paper"]["text"]
rows = ["| Metric | Persistence beyond the shuffled null (SE) |", "|---|---|"]
rows += [f"| {r[0]} | {r[4]} |" for r in T["rows"]]
text = "\n".join(rows) + (
    f"\n\nPersistence is the actual slope minus the mean slope over {X['d_n_shuffles']} within-hitter-season shuffles "
    f"(rows marked single shuffle use one). Bat-speed persistence averaged over the next 200 PA is still {A['decay_200']} "
    f"(about {A['weeks_200']} weeks for a regular). Out of sample, relative to the middle deciles of a swing-form index, the top "
    f"decile gained {A['top_swing']} (±{A['top_swing_se']}) wOBA points over the next 50 PA and the bottom decile lost "
    f"{A['bot_swing']} (±{A['bot_swing_se']}); {X['spec_bot_lo']}–{X['spec_bot_hi']} across the 50-PA, 100-PA and calendar windows, "
    f"adjustments, survivorship corrections and projection baselines, {X['w2525_bot']} with 25-PA windows. A plain wOBA streak: "
    f"bottom decile {A['res_bot']} (±{A['res_bot_se']}), top decile {A['top_results']} (±{A['top_results_se']}). Platoon "
    f"penalties in the same data: {A['lhb_pen']} wOBA points (LHB), {A['rhb_pen']} (RHB). Full results: `results/paper_results.json`.")
p = os.path.join(HERE, "README.md")
s = open(p).read()
s = re.sub(r"(<!-- headline:start -->\n).*?(<!-- headline:end -->)", lambda m: m.group(1) + text + "\n" + m.group(2), s, flags=re.S)
open(p, "w").write(s)
print(text)
