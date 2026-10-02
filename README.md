# How to Predict Hot Streaks in Major League Baseball: code and results

Code, public inputs and results for a study of hot streaks with Statcast bat tracking (SSAC27 research paper, baseball
track). Every regular-season pitch from 2024 to 2026 (Baseball Savant), 2.14 million pitches and 904 hitters. Every
number in the abstract and the paper is written from `results/paper_results.json`, and `verify_paper.py` checks them
with code written separately from the analysis.

## Reproduce

```
pip install pybaseball pyarrow pandas numpy statsmodels matplotlib python-docx
python fetch_statcast.py                                   # monthly parquet files in statcast_hotstreak/, resumable
python hot_streak_study.py statcast_hotstreak out_main     # panel, windows, shuffled null, decile tables
python analysis2.py       statcast_hotstreak out_main      # park/week adjustment, forecast, leave-one-season-out index
python simulate.py sim_hot 1.0 && python hot_streak_study.py sim_hot out_sim_hot    # pipeline check: injected state
python simulate.py sim_null 0  && python hot_streak_study.py sim_null out_sim_null  # pipeline check: no state

# full-paper analyses (plans: PLAN_2026-09-30.md and addenda B to E, each written and hashed before the analyses ran)
python fetch_public.py statcast_hotstreak data_public            # IL transactions, Marcel inputs, postseason (MLB Stats API)
FG_BAT=path/to/fangraphs_position_player_lines.xlsx SHARED_STATCAST=path/to/full_width_statcast_by_season \
python analysis3.py statcast_hotstreak data_public out3          # A1-A10 -> out3/a1.json ... a10.json
python null_mc.py out3                                            # 20-shuffle null
python make_paper_results.py statcast_hotstreak out3 out_main/results2.json out3/paper_results.json
python export_frame.py out3 && python horizon_weeks.py out3
FG_BAT=... python analysis4_reversion.py statcast_hotstreak data_public out3 real sims merge   # A11, offseason (exploratory)
python b3_a7_pairs.py statcast_hotstreak data_public out3         # lineup decisions per pair (addendum D)
python paper_counts.py statcast_hotstreak out3                    # descriptive counts for the paper
python e_review.py statcast_hotstreak data_public out3            # checks added after review (addendum E, exploratory)
cp out_main/results2.json out3/results_adjusted.json
python make_paper_block.py out3 [companion_results.json]          # every number the paper prints -> paper_results.json['paper']
python make_figures.py out3/paper_results.json figures
FG_BAT=... python verify_paper.py statcast_hotstreak out3/paper_results.json out3/forecast_frame_50_50.parquet \
    path/to/abstract.md [path/to/paper.docx]                      # must end with ALL CHECKS PASSED
```

`fetch_statcast.py` and `fetch_public.py` need network access to baseballsavant.mlb.com and statsapi.mlb.com; everything
else runs offline. The FanGraphs season lines (projection baseline and survivorship weights) are not redistributed;
download them from FanGraphs. `make_paper_block.py` stores the two companion-paper values it quotes, with their source
key and file hash, so later runs do not need the companion file.

## What it does

* `hot_streak_study.py` builds a batter-game panel, then for every hitter-game compares the previous N plate appearances
  with the next M on 16 metrics (results, contact quality, swing decisions, swing mechanics). **Persistence** is the
  slope of next-window deviation on recent deviation, both measured against the rest of that hitter's season, **minus
  the same slope on seasons whose game order was shuffled within hitter**. The shuffle keeps talent, playing time and
  the season's opponents and parks and removes time structure, so what is left is carry-over beyond what the estimator
  produces on its own. Persistence is a slope, so it is attenuated by noise in the recent window.
* `analysis2.py` adds park-season and week-of-season adjustments for bat speed, swing length and exit velocity, a
  forecast of next-50-PA wOBA from history, season to date and forward-window context, and a leave-one-season-out
  swing-form index, with hitter-clustered standard errors.
* `analysis3.py` runs A1-A10 (decay, survivorship, projection baselines, heterogeneity, pitchers, postseason, lineup
  decisions, injured list, literature reconciliation, robustness). `analysis4_reversion.py` runs the offseason
  analysis A11 with calibration simulations from `simulate.py`.
* `make_paper_block.py` formats every number the paper prints, `make_figures.py` draws the figures in `figures/`, and
  `verify_paper.py` recomputes the headline estimates independently and scans the abstract and the paper for any number
  not in `paper_results.json`.

## Headline numbers (50 PA → next 50 PA, park/week-adjusted; generated from `results/paper_results.json`)

<!-- headline:start -->
| Metric | Persistence beyond the shuffled null (SE) |
|---|---|
| Bat speed | 26.0% (1.5) |
| Swing length | 25.3% (1.7) |
| Fast-swing rate | 22.8% (1.8) |
| In-zone swing rate | 12.2% (1.4) |
| Chase rate | 7.4% (1.4) |
| Whiff rate | 2.0% (1.4) |
| Exit velocity | 2.3% (1.4) |
| Hard-hit rate | −0.1% (1.4) |
| Barrel rate | 4.0% (1.5) |
| Strikeout rate | 4.3% (1.4) |
| xwOBA | 2.0% (1.4) |
| wOBA | 0.3% (1.4) |
| Squared-up rate (single shuffle) | 2.8% (1.4) |
| Blast rate (single shuffle) | 2.4% (1.4) |
| xwOBA on contact (single shuffle) | 2.6% (1.4) |
| Walk rate (single shuffle) | 2.1% (1.5) |

Persistence is the actual slope minus the mean slope over 20 within-hitter-season shuffles (rows marked single shuffle use one). Bat-speed persistence averaged over the next 200 PA is still 14.2% (about 9 weeks for a regular). Out of sample, relative to the middle deciles of a swing-form index, the top decile gained 8.4 (±3.2) wOBA points over the next 50 PA and the bottom decile lost 7.9 (±2.9); 5.7–9.4 across the 50-PA, 100-PA and calendar windows, adjustments, survivorship corrections and projection baselines, −3.6 with 25-PA windows. A plain wOBA streak: bottom decile −1.1 (±2.6), top decile 3.4 (±2.6). Platoon penalties in the same data: 26.5 wOBA points (LHB), 10.4 (RHB). Full results: `results/paper_results.json`.
<!-- headline:end -->

## Notes and limits

* Among decision points with a complete forward window, a slowing bat was not associated with an injured-list placement
  within 30 days (OR 0.96 per −1 SD of recent bat speed, 0.87–1.07, any placement, labels from the public transaction
  feed; A8, `paper_results.json['abstract']['il_or']`). The window requirement drops points just before season-ending
  placements.
* Bat tracking is missing for the first week of 2024. The 2026 season was pulled on 2026-09-30, after the regular season
  ended and before Savant's off-season reprocessing.
* Persistence is attenuated by measurement noise, and 50-PA wOBA is very noisy, so the near-zero wOBA persistence does
  not show that hitters lack a results-level state. It shows that the swing measures a state better.
* Most bat-speed persistence is within-season shifts of level, which can come from form, equipment, deliberate swing
  changes, playing hurt or tracking. The offseason analysis (A11) was undetermined under its prespecified rule.
* A hitter needs 50 more PA that season to be a data point. Three survivorship corrections leave the cold tail in place
  but none models benching during the forward window.
* The forecast uses the opponents and parks actually faced in the forward window and full-season opponent quality, and
  it leaves the recent 50 PA out of the hitter's history. The lineup run values are model-expected.
* The decile effects vary by season (2024's bottom decile is positive) and are smaller with 25-PA forward windows; only
  the top-bottom spread replicates every season.
* Bat-speed persistence depends on the competitive-swing filter (13.3% without it); swing length, fast-swing rate and the
  decile effects do not.
* Regular season only. The postseason check is uninformative.
