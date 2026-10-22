#!/usr/bin/env bash
# Final run (PLAN_ADDENDUM_2026-10-17h.md, item 14; author items of 2026-10-21, post hoc): the IL stints of 2026-10-20 with
#   (1) the 1 stint into which a record naming a different injury was merged classified by that later injury (a hamate
#       fracture placed on the 60-day list during a stint for a foot contusion; ojsm_il_final_2026-10-21.py),
#   (2) the 2 rib fractures coded chest or rib instead of oblique/abdominal (body region of Table 1 only; same script), and
#   (3) the activations missing from the IL file added, each dated by the hitter's first regular-season game after the
#       placement (ojsm_missing_activations.py; a sensitivity analysis in the run of 2026-10-20, the primary analysis here).
# Every analysis that reads an IL file is rerun into repo/results/final_2026-10-21/ (same file names as in
# repo/results/stints_2026-10-20/, which keeps the run with the recorded activations only, now the sensitivity analysis).
#
#   bash repo/run_ojsm_final_2026-10-21.sh STATCAST_DIR STATSAPI_DIR          (from the Hot_Hand_Bat_Tracking folder, after
#                                                                             repo/run_ojsm_stints_2026-10-20.sh)
#
# STATCAST_DIR: the original season files (MLB Total Research/Statcast_Shared/data/statcast_season, 2024-2026).
# STATSAPI_DIR: transactions_2024-01-01_2026-10-05.json and schedule_2024_2026.json (repo/data_public/statsapi/, gunzipped).
set -euo pipefail
SC="$1"; SAPI="$2"
DP=repo/data_public; ST=repo/results/stints_2026-10-20; FIN=repo/results/final_2026-10-21; FG=inputs_fangraphs_not_redistributed
CODED=$DP/il_placements_coded_2023_2026.csv
CODEDF=$DP/il_placements_coded_2023_2026_final.csv
ILF=$DP/il_transactions_hitters_2024_2026_final.txt            # corrected dates, stints, (1) and (3): the primary IL file
ILO=$DP/il_transactions_hitters_2024_2026_stints_final.txt     # record dates, stints, (1): the prespecified season-level file
mkdir -p "$FIN"
python3 repo/ojsm_il_final_2026-10-21.py $ST/ojsm_il_stints.json $DP/il_transactions_hitters_2024_2026_corrected_stints.txt \
    $DP/il_transactions_hitters_2024_2026_stints.txt "$CODED" $FIN/il_transactions_hitters_2024_2026_corrected_stints_relabeled.txt \
    $ILO "$CODEDF" $FIN/ojsm_il_final.json > $FIN/ojsm_il_final_run.log 2>&1
python3 repo/ojsm_missing_activations.py $FIN/il_transactions_hitters_2024_2026_corrected_stints_relabeled.txt "$SC" \
    "$SAPI/transactions_2024-01-01_2026-10-05.json" $ILF $FIN/ojsm_missing_activations.json > $FIN/ojsm_missing_activations_run.log 2>&1
python3 repo/ajsm_season.py $ILO "$FG/Fangraphs Bat Tracking ALL.csv" \
    "$FG/Fangraphs Position Player Raw 2000-2025.xlsx" "$FG/fangraphs_batting_2026.csv" $FIN/ajsm_season.json > $FIN/ajsm_season_run.log 2>&1
python3 repo/ajsm_season.py $ILF "$FG/Fangraphs Bat Tracking ALL.csv" \
    "$FG/Fangraphs Position Player Raw 2000-2025.xlsx" "$FG/fangraphs_batting_2026.csv" $FIN/ajsm_season_corrected.json "$CODEDF" > $FIN/ajsm_season_corrected_run.log 2>&1
python3 repo/ajsm_game_ojsm.py "$SC" $ILF $FIN/ajsm_game_ojsm.json \
    --shuffles 5 --coded "$CODEDF" --il-original $ILO > $FIN/ajsm_game_run.log 2>&1
python3 repo/ojsm_new_analyses.py "$SC" $ILF "$CODEDF" "$SAPI" $FIN $FIN/ojsm_new_analyses.json --shuffles 5 > $FIN/ojsm_new_analyses_run.log 2>&1
python3 repo/ojsm_new_analyses_b.py $FIN "$CODEDF" > $FIN/ojsm_new_analyses_b_run.log 2>&1
python3 repo/ojsm_control_draws.py "$SC" $ILF "$CODEDF" $ILO $FIN/ojsm_control_draws.json --draws 1000 > $FIN/ojsm_control_draws_run.log 2>&1
python3 repo/ojsm_flow_counts.py "$SC" $ILF "$CODEDF" \
    $DP/il_transactions_described_2023_2026_raw.txt $DP/il_transactions_hitters_2024_2026.txt $FIN/ojsm_flow_counts.json > $FIN/ojsm_flow_counts_run.log 2>&1
python3 repo/ajsm_reconcile_return.py "$SC" $ILF $FIN/ajsm_reconcile_return.json > $FIN/ajsm_reconcile_return_run.log 2>&1
python3 repo/ojsm_stints_compare.py repo/results $FIN $FIN/ojsm_stints_compare.json \
    il_transactions_hitters_2024_2026_stints_final.txt il_transactions_hitters_2024_2026_final.txt > $FIN/ojsm_stints_compare_run.log 2>&1
python3 repo/ojsm_season_ending.py $ILF "$CODEDF" "$FG/Fangraphs Position Player Raw 2000-2025.xlsx" \
    "$FG/fangraphs_batting_2026.csv" "$SC" $FIN/ajsm_season_corrected.json $FIN/ojsm_season_ending.json > $FIN/ojsm_season_ending_run.log 2>&1
python3 repo/ojsm_f4b_inclusion.py "$SC" $ILF "$CODEDF" $FIN $FIN/ojsm_f4b_inclusion.json > $FIN/ojsm_f4b_inclusion_run.log 2>&1
python3 repo/ojsm_season_cohort_changes.py $ILO "$FG/Fangraphs Bat Tracking ALL.csv" \
    "$FG/Fangraphs Position Player Raw 2000-2025.xlsx" "$FG/fangraphs_batting_2026.csv" $FIN/ajsm_season.json $FIN/ojsm_stints_compare.json \
    $FIN/ojsm_season_cohort_changes.json > $FIN/ojsm_season_cohort_changes_run.log 2>&1
# Table 1 and Results: prior-season PA of the season-level cohort (quartiles; ajsm_season.py stores the median only) and the
# 2024 placement rate in MLB's complete transaction log
python3 repo/ojsm_f1_pa_iqr.py $ILO "$FG/Fangraphs Bat Tracking ALL.csv" "$FG/Fangraphs Position Player Raw 2000-2025.xlsx" \
    "$FG/fangraphs_batting_2026.csv" $FIN/ajsm_season.json $FIN/ojsm_f1_pa_iqr.json > $FIN/ojsm_f1_pa_iqr_run.log 2>&1
python3 repo/ojsm_placement_rate_check.py "$SAPI/transactions_2024-01-01_2026-10-05.json" $DP/il_transactions_hitters_2024_2026_corrected.txt \
    $FIN/ojsm_placement_rate_check.json > $FIN/ojsm_placement_rate_check_run.log 2>&1
# the IL rows of Supplemental Table S5 (original injury analysis, not rerun): observations the added activations could move
python3 repo/ojsm_s5_activation_bound.py $FIN/ojsm_missing_activations.json $DP/il_transactions_hitters_2024_2026.txt "$SC" \
    repo/results/paper_results.json $FIN/ojsm_s5_activation_bound.json > $FIN/ojsm_s5_activation_bound_run.log 2>&1
# tracked and competitive swings counted on the same basis (Methods, Table 1; independent check of 2026-10-21)
python3 repo/ojsm_swing_counts.py "$SC" $FIN/ojsm_swing_counts.json > $FIN/ojsm_swing_counts_run.log 2>&1
echo "done: $FIN"
