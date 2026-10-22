#!/usr/bin/env bash
# Author decision of 2026-10-20 (PLAN_ADDENDUM_2026-10-17h.md, item 12): every analysis on IL stints rather than IL placement
# records. Writes the merged IL files and reruns every analysis that reads an IL file into repo/results/stints_2026-10-20/
# (same file names as in repo/results/, which keeps the runs on the placement records, including the reproduction of the
# AJSM values that build_ojsm.py checks).
#
#   bash repo/run_ojsm_stints_2026-10-20.sh STATCAST_DIR STATSAPI_DIR          (from the Hot_Hand_Bat_Tracking folder)
#
# STATCAST_DIR: the original season files (MLB Total Research/Statcast_Shared/data/statcast_season, 2024-2026).
# STATSAPI_DIR: transactions_2024-01-01_2026-10-05.json and schedule_2024_2026.json (repo/data_public/statsapi/, gunzipped).
set -euo pipefail
SC="$1"; SAPI="$2"
DP=repo/data_public; ST=repo/results/stints_2026-10-20; FG=inputs_fangraphs_not_redistributed
CODED=$DP/il_placements_coded_2023_2026.csv
mkdir -p "$ST"
python3 repo/ojsm_il_stints.py $DP/il_transactions_hitters_2024_2026.txt $DP/il_transactions_hitters_2024_2026_corrected.txt "$CODED" \
    $DP/il_transactions_hitters_2024_2026_stints.txt $DP/il_transactions_hitters_2024_2026_corrected_stints.txt $ST/ojsm_il_stints.json
python3 repo/ajsm_season.py $DP/il_transactions_hitters_2024_2026_stints.txt "$FG/Fangraphs Bat Tracking ALL.csv" \
    "$FG/Fangraphs Position Player Raw 2000-2025.xlsx" "$FG/fangraphs_batting_2026.csv" $ST/ajsm_season.json > $ST/ajsm_season_run.log 2>&1
python3 repo/ajsm_season.py $DP/il_transactions_hitters_2024_2026_corrected_stints.txt "$FG/Fangraphs Bat Tracking ALL.csv" \
    "$FG/Fangraphs Position Player Raw 2000-2025.xlsx" "$FG/fangraphs_batting_2026.csv" $ST/ajsm_season_corrected.json "$CODED" > $ST/ajsm_season_corrected_run.log 2>&1
python3 repo/ajsm_game_ojsm.py "$SC" $DP/il_transactions_hitters_2024_2026_corrected_stints.txt $ST/ajsm_game_ojsm.json \
    --shuffles 5 --coded "$CODED" --il-original $DP/il_transactions_hitters_2024_2026_stints.txt > $ST/ajsm_game_run.log 2>&1
python3 repo/ojsm_new_analyses.py "$SC" $DP/il_transactions_hitters_2024_2026_corrected_stints.txt "$CODED" "$SAPI" $ST \
    $ST/ojsm_new_analyses.json --shuffles 5 > $ST/ojsm_new_analyses_run.log 2>&1
python3 repo/ojsm_new_analyses_b.py $ST "$CODED" > $ST/ojsm_new_analyses_b_run.log 2>&1
python3 repo/ojsm_control_draws.py "$SC" $DP/il_transactions_hitters_2024_2026_corrected_stints.txt "$CODED" \
    $DP/il_transactions_hitters_2024_2026_stints.txt $ST/ojsm_control_draws.json --draws 1000 > $ST/ojsm_control_draws_run.log 2>&1
# Figure 1 and the IL check: the exclusion steps on the stint file; the IL check compares the record-date file as used before
# the merge with MLB's records (Supplemental Table S4)
python3 repo/ojsm_flow_counts.py "$SC" $DP/il_transactions_hitters_2024_2026_corrected_stints.txt "$CODED" \
    $DP/il_transactions_described_2023_2026_raw.txt $DP/il_transactions_hitters_2024_2026.txt $ST/ojsm_flow_counts.json > $ST/ojsm_flow_counts_run.log 2>&1
python3 repo/ajsm_reconcile_return.py "$SC" $DP/il_transactions_hitters_2024_2026_corrected_stints.txt $ST/ajsm_reconcile_return.json \
    > $ST/ajsm_reconcile_return_run.log 2>&1
python3 repo/ojsm_stints_compare.py repo/results $ST $ST/ojsm_stints_compare.json
# third independent check (2026-10-20): Table 1 season-ending placements and stint lengths from the games played and the end
# of the regular season (also on placement records, for Supplemental Table S13); activations missing from the IL file and the
# after-return analysis rerun with them; the rows of the every-eligible-control analysis; the season-level cohorts of the
# hitter-seasons whose families changed
R3=$ST/recheck3; mkdir -p "$R3/missing_activations"
python3 repo/ojsm_season_ending.py $DP/il_transactions_hitters_2024_2026_corrected_stints.txt "$CODED" "$FG/Fangraphs Position Player Raw 2000-2025.xlsx" \
    "$FG/fangraphs_batting_2026.csv" "$SC" $ST/ajsm_season_corrected.json $R3/ojsm_season_ending.json
python3 repo/ojsm_season_ending.py $DP/il_transactions_hitters_2024_2026_corrected.txt "$CODED" "$FG/Fangraphs Position Player Raw 2000-2025.xlsx" \
    "$FG/fangraphs_batting_2026.csv" "$SC" repo/results/ajsm_season_corrected.json $R3/ojsm_season_ending_records.json
python3 repo/ojsm_missing_activations.py $DP/il_transactions_hitters_2024_2026_corrected_stints.txt "$SC" "$SAPI/transactions_2024-01-01_2026-10-05.json" \
    $R3/il_transactions_hitters_2024_2026_corrected_stints_activations_added.txt $R3/ojsm_missing_activations.json
python3 repo/ajsm_game_ojsm.py "$SC" $R3/il_transactions_hitters_2024_2026_corrected_stints_activations_added.txt $R3/missing_activations/ajsm_game_ojsm.json \
    --shuffles 5 --coded "$CODED" > $R3/missing_activations/ajsm_game_run.log 2>&1
python3 repo/ojsm_f4b_inclusion.py "$SC" $DP/il_transactions_hitters_2024_2026_corrected_stints.txt "$CODED" $ST $R3/ojsm_f4b_inclusion.json \
    > $R3/ojsm_f4b_inclusion_run.log 2>&1
python3 repo/ojsm_season_cohort_changes.py $DP/il_transactions_hitters_2024_2026_stints.txt "$FG/Fangraphs Bat Tracking ALL.csv" \
    "$FG/Fangraphs Position Player Raw 2000-2025.xlsx" "$FG/fangraphs_batting_2026.csv" $ST/ajsm_season.json $ST/ojsm_stints_compare.json \
    $R3/ojsm_season_cohort_changes.json
echo "done: $ST"
