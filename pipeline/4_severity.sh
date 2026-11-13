#!/usr/bin/env bash
# Table 3 (Square under weaker perturbations): development sweep of milder strengths (seed 0, 20
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
probe_out=$O/repairable_dpsq_mild_abs.txt
[ -f $probe_out ] || python scripts/interp_repairable.py --base "$N/dpsq_cf1_none" \
  --shift "wroll90=$N/dpsq_cf1_wroll90" --shift "faint03=$N/dpsq_cf1_faint03" --absolute > $probe_out

python scripts/interp_grid.py --table severity
