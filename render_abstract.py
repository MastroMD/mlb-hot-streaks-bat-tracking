"""Fill sloan/SSAC27_abstract.template.md from paper_results.json['abstract'] and report the word count.
python3 render_abstract.py paper_results.json TEMPLATE OUT_MD
Word count follows the SSAC rule (title + body, fewer than 500); the table is reported separately."""
import json
import re
import sys

R = json.load(open(sys.argv[1]))["abstract"]
R = dict(R)
R["extreme"] = f"{round((float(R['top_swing']) + float(R['bot_swing'])) / 2):.0f}"
t = open(sys.argv[2]).read()
missing = sorted(set(re.findall(r"\{(\w+)\}", t)) - set(R))
if missing:
    sys.exit(f"template keys missing from paper_results.json: {missing}")
out = re.sub(r"\{(\w+)\}", lambda m: R[m.group(1)], t)
open(sys.argv[3], "w").write(out)

body, _, rest = out.partition("**Figure 1." if "**Figure 1." in out else "**Table 1.")
table, _, concl = rest.partition("## Conclusion")
wc = lambda s: len(re.sub(r"[#*|]", " ", s).split())
n_text = wc(body) + wc("## Conclusion" + concl) - wc("**Track:** Baseball")
print(f"words (title + body, excluding the track line and the table or figure caption): {n_text}")
print(f"words in the table/figure caption: {wc('**Table 1.' + table)};  whole file: {wc(out)}")
if n_text >= 500:
    sys.exit("OVER the SSAC limit (fewer than 500 words)")
