#!/usr/bin/env bash
# Table 2 (object position on Lift) and the probe and swap numbers beside it. Every arm is
# evaluated on fresh scenes (env seed 3): the cube offset ±5/±7 cm along x and y (40 episodes each;
# SmolVLA 30) and unshifted (100; SmolVLA 50). Labels are collected on training scenes (seeds 1, 2).
source "$(dirname "$0")/common.sh"
# label offsets: 0,0, ±3/±5/±7 cm on x and y
ALL13='0,0 -0.07,0 -0.05,0 -0.03,0 0.03,0 0.05,0 0.07,0 0,-0.07 0,-0.05 0,-0.03 0,0.03 0,0.05 0,0.07'
X6='-0.07,0 -0.05,0 -0.03,0 0.03,0 0.05,0 0.07,0'; Y6='0,-0.07 0,-0.05 0,-0.03 0,0.03 0,0.05 0,0.07'

evals() {   # name backend weights [episodes at an offset]: the arm on every offset, seed 3
  local eps=${4:-40}
  col $O/lift_$1_s3_shift_0,0 lift $2 $3 3 $(( eps == 30 ? 50 : 100 ))
  for s in $OFFSETS; do col $O/lift_$1_s3_shift_$s lift $2 $3 3 $eps --shift=$s; done
}

# probe and interchange: decodability of the cube offset at each ACT site and the action's gain
# (S_probe, S_act), and the swap test (out-of-range feature directions vs the head)
[ -f $O/sites_lift_act_base.npz ] || python scripts/probe_sites.py --weights $ACT_LIFT \
  --out $O/sites_lift_act_base.npz > $O/sites_lift_act_base.log 2>&1

evals act_base act $ACT_LIFT
evals act_xaug act $ACT_LIFT_X            # full retrain, 400 x-shifted demos
evals act_yaug act $ACT_LIFT_Y

# head retrain, DAgger expert labels: the base drives (seed 1) and the coverage retrain of that
# axis labels; round 2: the round-1 edit drives (seed 2); the head refit on both rounds
dagger() {   # axis teacher shifts
  local a=$1 T=$2 s
  for s in 0,0 $3; do col $D/r1_${a}_$s lift act $ACT_LIFT 1 20 --shift=$s --teacher $T; done
  edit $E/${a}_onpolicy_head $ACT_LIFT --onpolicy "$D/r1_${a}_*" --rank 10
  for s in 0,0 $3; do col $D/r2_${a}_$s lift act $E/${a}_onpolicy_head 2 20 --shift=$s --teacher $T; done
  edit $E/${a}_onpolicy_head_r2 $ACT_LIFT --onpolicy "$D/r1_${a}_*,$D/r2_${a}_*" --rank 10
  evals act_edit_${a}_onpolicy_head_r2 act $E/${a}_onpolicy_head_r2
}
dagger x $ACT_LIFT_X "$X6"
dagger y $ACT_LIFT_Y "$Y6"

# head retrain on pseudo-labels, no demos: the policy's own chunk with the cube moved to the
# demos' box edge, extended to the real cube by its local slope rescaled to gain 1
for s in $ALL13; do col $D/sl2_$s lift act $ACT_LIFT 1 20 --shift=$s --selflabel full; done
edit $E/slfull_b3 $ACT_LIFT --onpolicy "$D/sl2_*" --rank 10 --beta 3
evals act_edit_slfull_b3 act $E/slfull_b3

# control: the same head retrain on in-distribution labels (unshifted demo replays)
edit $E/x_head_r10_b1_unsh $ACT_LIFT --site head --rank 10 --beta 1 --labels unshifted
evals act_edit_x_head_r10_b1_unsh act $E/x_head_r10_b1_unsh

# SmolVLA: base, and its action head (action_out_proj) retrained per axis on the coverage
# retrain's labels of its own states (seed 1)
evals vla_base smolvla $VLA 30
vla_axis() {   # axis teacher shifts
  local s
  for s in 0,0 $3; do
    col $O/vla/${1}t_$s lift smolvla $VLA 1 10 --shift=$s --teacher $2 --label-every 2 --save-images
  done
  [ -f $O/vla/rep_${1}pos/edit.json ] || python scripts/repair_vla.py --parent $VLA \
    --out $O/vla/rep_${1}pos --data "$O/vla/${1}t_*" --steps 3000 > $O/vla/rep_${1}pos.log 2>&1
  evals vla_rep_${1}pos smolvla $O/vla/rep_${1}pos 30
}
vla_axis x $ACT_LIFT_X "$X6"
vla_axis y $ACT_LIFT_Y "$Y6"
# pseudo-labels (fixed sampling noise) and the control on in-distribution labels, both axes at once
vla_refit() { [ -f $O/vla/$1/edit.json ] || python scripts/repair_vla.py --parent $VLA --out $O/vla/$1 \
  --data "$2" > $O/vla/$1.log 2>&1; }
for s in $ALL13; do
  col $O/vla/sl_$s lift smolvla $VLA 1 10 --shift=$s --selflabel full --label-every 2 --save-images
done
col $O/vla/pos_cf1_none lift smolvla $VLA 1 20 --selflabel cf --label-every 2 --save-images
vla_refit rep_slfull "$O/vla/sl_*"
vla_refit rep_posctl "$O/vla/pos_cf1_none"
evals vla_slfull smolvla $O/vla/rep_slfull 30
evals vla_posctl smolvla $O/vla/rep_posctl 30

# Diffusion Policy: base; its FiLM readout retrained per axis on the ACT expert's labels (two
# DAgger rounds, as ACT's), on pseudo-labels (fixed noise), and on in-distribution labels (control)
for s in $OFFSETS; do col $O/lift_dp_base_s3_shift_$s lift dp $DP_LIFT 3 40 --shift=$s; done
dp_refit() { [ -f $1 ] || python scripts/repair_dp.py --parent $DP_LIFT --out $1 --data "$2" > $1.log 2>&1; }
dp_evals() {   # name ckpt offsets: 100 unshifted, 40 per offset
  local s
  col $O/lift_$1_s3_shift_0,0 lift dp $2 3 100
  for s in $3; do col $O/lift_$1_s3_shift_$s lift dp $2 3 40 --shift=$s; done
}
dp_dagger() {   # axis teacher shifts evals
  local a=$1 T=$2 s
  for s in 0,0 $3; do col $D/dp_r1_${a}_$s lift dp $DP_LIFT 1 20 --shift=$s --teacher $T; done
  dp_refit $E/dppos_${a}_r1.ckpt "$D/dp_r1_${a}_*"
  for s in 0,0 $3; do col $D/dp_r2_${a}_$s lift dp $E/dppos_${a}_r1.ckpt 2 20 --shift=$s --teacher $T; done
  dp_refit $E/dppos_${a}_r2.ckpt "$D/dp_r1_${a}_*,$D/dp_r2_${a}_*"
  dp_evals dp_pos${a}_r2 $E/dppos_${a}_r2.ckpt "$4"
}
dp_dagger x $ACT_LIFT_X "$X6" '-0.07,0 -0.05,0 0.05,0 0.07,0'
dp_dagger y $ACT_LIFT_Y "$Y6" '0,-0.07 0,-0.05 0,0.05 0,0.07'
for s in $ALL13; do col $D/dpsl_$s lift dp $DP_LIFT 1 20 --shift=$s --selflabel full; done
dp_refit $E/dppos_sl.ckpt "$D/dpsl_*"
dp_evals dp_possl $E/dppos_sl.ckpt "$OFFSETS"
col $N/dp_cf1_none lift dp $DP_LIFT 1 30 --selflabel cf --label-every 4      # also 3_perturb.sh's
dp_refit $E/dppos_ctl.ckpt "$N/dp_cf1_none"
dp_evals dp_posctl $E/dppos_ctl.ckpt "$OFFSETS"

python scripts/tables.py --table position
