#!/usr/bin/env bash
# Task or severity (Square under weaker perturbations; a paragraph in the paper): development sweep of milder strengths (seed 0, 20
# episodes), then the full head-retrain protocol (3_perturb.sh's) on the partial failures.
source "$(dirname "$0")/common.sh"
SPEC[faint03]='objalpha=0.3'; SPEC[faint05]='objalpha=0.5'; SPEC[wroll90]='wroll=90'
SPEC[cam10w30]='cam=0,0,0,10;wroll=30'; SPEC[light05]='light=0.5'

for t in none faint faint03 faint05 wroll90 cam10w30 light05; do
  s=(); [ $t = none ] || s=(--nuisance "${SPEC[$t]}")
  col $N/sqbase_$t square act $ACT_SQ 0 20 "${s[@]}"
  col $N/dpsqbase_$t square dp $DP_SQ 0 20 "${s[@]}"
done

for t in faint03 faint05; do act_protocol square $ACT_SQ sq_ 30 40 100 $t; done
for t in faint03 wroll90; do dp_protocol square $DP_SQ dpsq_ $t; done
[ -f $O/repairable_sq_mild_abs.txt ] || python scripts/probe_r2.py --base "$N/sq_cf1_none" \
  --shift "faint03=$N/sq_cf1_faint03" --shift "faint05=$N/sq_cf1_faint05" --absolute --stride 2 \
  > $O/repairable_sq_mild_abs.txt
probe_out=$O/repairable_dpsq_mild_abs.txt
[ -f $probe_out ] || python scripts/probe_r2.py --base "$N/dpsq_cf1_none" \
  --shift "wroll90=$N/dpsq_cf1_wroll90" --shift "faint03=$N/dpsq_cf1_faint03" --absolute --stride 2 > $probe_out

python scripts/tables.py --table severity
