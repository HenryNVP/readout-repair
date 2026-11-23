#!/usr/bin/env bash
# Table 2 (visual perturbations across policies and tasks) and the probe R² beside it; the
# per-cell protocol is common.sh's act_protocol / dp_protocol / vla_protocol.
source "$(dirname "$0")/common.sh"

for t in bothcam darkredwroll bothflip bothzoom faint invisible; do act_protocol lift $ACT_LIFT '' 30 60 100 $t; done
# 200 more fresh-scene episodes (seed 4) for the cell closest to the bar and its control
col $N/eval4_rep_bothcam lift act $E/rep_bothcam 4 200 --nuisance "${SPEC[bothcam]}"
col $N/eval4_rep_invisible lift act $E/rep_invisible 4 200 --nuisance "${SPEC[invisible]}"
for t in bothcam darkredwroll bothflip faint invisible; do dp_protocol lift $DP_LIFT dp_ $t; done
for t in bothcam darkredwroll bothflip faint invisible; do vla_protocol $t; done
for t in bothcam darkredwroll bothflip bothzoom faint invisible; do act_protocol square $ACT_SQ sq_ 30 40 100 $t; done
for t in bothcam darkredwroll bothflip faint invisible; do dp_protocol square $DP_SQ dpsq_ $t; done
for t in bothflip invisible; do act_protocol toolhang $ACT_TH th_ 20 30 50 $t; done

# in-distribution base success, the failure rule's reference (Lift ACT: 2_position.sh's unshifted)
col $N/dp_eval3_base_none lift dp $DP_LIFT 3 100
col $N/vla_eval3_base_none lift smolvla $VLA 3 50
col $N/sq_eval3_base_none square act $ACT_SQ 3 100
col $N/dpsq_eval3_base_none square dp $DP_SQ 3 100
col $N/th_eval3_base_none toolhang act $ACT_TH 3 50

# probe: is the grasp point linearly decodable at the head input (before contact)?
probe() {   # out prefix tags...
  local out=$1 X=$2 a=() t
  shift 2
  for t in "$@"; do a+=(--shift "$t=$N/${X}cf1_$t"); done
  [ -f $out ] || python scripts/probe_r2.py --base "$N/${X}cf1_none" "${a[@]}" --absolute --stride 2 > $out
}
probe $O/repairable_abs.txt '' bothcam darkredwroll bothflip bothzoom faint invisible
probe $O/repairable_dp_abs.txt dp_ bothcam darkredwroll bothflip faint invisible
probe $O/repairable_sq_abs.txt sq_ bothcam darkredwroll bothflip bothzoom faint invisible
probe $O/repairable_dpsq_abs.txt dpsq_ bothcam darkredwroll bothflip faint invisible
probe $O/repairable_th_abs.txt th_ bothflip invisible

python scripts/tables.py --table perturb
