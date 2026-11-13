#!/usr/bin/env bash
# Figure 3 (Square needs depth; ACT, turn + roll): DAgger iterations of full fine-tuning, then the
# head, LoRA (rank 8, every linear layer) and full fine-tuning on the same four iterations of
# labels; the forgetting check with 100 demonstrations replayed; ToolHang's two iterations.
# Iteration k: the iteration-(k-1) policy drives, the unedited policy labels the shift-free render.
source "$(dirname "$0")/common.sh"
S=${SPEC[bothcam]}
fit() {   # name data [interp_edit args...]
  local n=$1 data=$2
  shift 2
  edit $E/$n $ACT_SQ --images "$data" --pres none --beta 0 --batch 32 "$@"
  col $N/${n}_eval3_bothcam square act $E/$n 3 100 --nuisance "$S"
  col $N/${n}_eval3_none square act $E/$n 3 100
}
FULL='--site full --lr 3e-5'
c() { col $N/sqimg_$1 square act $3 $2 $4 --selflabel cf --save-images "${@:5}"; }   # name seed weights n [args]

c cf1_none 1 $ACT_SQ 30
c cf1_bothcam 1 $ACT_SQ 40 --nuisance "$S"
R1="$N/sqimg_cf1_bothcam,$N/sqimg_cf1_none"
fit sq_depth_full "$R1" $FULL --steps 3000
c cf2_bothcam 2 $E/sq_depth_full 40 --nuisance "$S" --label-weights $ACT_SQ
R2="$N/sqimg_cf1_bothcam,$N/sqimg_cf2_bothcam,$N/sqimg_cf1_none"
fit sq_depth_full_r2 "$R2" $FULL --steps 5000
c cf3_bothcam 4 $E/sq_depth_full_r2 40 --nuisance "$S" --label-weights $ACT_SQ
c cf5_none 5 $ACT_SQ 30                                    # more in-range labels
R3="$N/sqimg_cf1_bothcam,$N/sqimg_cf2_bothcam,$N/sqimg_cf3_bothcam,$N/sqimg_cf1_none,$N/sqimg_cf5_none"
fit sq_depth_full_r3 "$R3" $FULL --steps 6000
c cf6_bothcam 6 $E/sq_depth_full_r3 40 --nuisance "$S" --label-weights $ACT_SQ
R4="$N/sqimg_cf1_bothcam,$N/sqimg_cf2_bothcam,$N/sqimg_cf3_bothcam,$N/sqimg_cf6_bothcam,$N/sqimg_cf1_none,$N/sqimg_cf5_none"
fit sq_depth_full_r4 "$R4" $FULL --steps 6000
c cf7_bothcam 7 $E/sq_depth_full_r4 40 --nuisance "$S" --label-weights $ACT_SQ
R5="$N/sqimg_cf1_bothcam,$N/sqimg_cf2_bothcam,$N/sqimg_cf3_bothcam,$N/sqimg_cf6_bothcam,$N/sqimg_cf7_bothcam,$N/sqimg_cf1_none,$N/sqimg_cf5_none"
fit sq_depth_full_r5 "$R5" $FULL --steps 6000

# depth on the same four iterations of labels
fit sq_depth_head_r4 "$R4" --site head --rank 10 --lr 1e-3 --steps 6000
fit sq_depth_all8_r4 "$R4" --site all --rank 8 --lr 1e-3 --steps 6000

# forgetting: the iteration-4 full fine-tune with 100 original demonstrations replayed
[ -f $N/sq_demos/ep0099.npz ] || python scripts/interp_demos_npz.py --dataset data/robomimic/square/ph/image.hdf5 \
  --out $N/sq_demos --max-demos 100 > $N/sq_demos.log 2>&1
fit sq_depth_full_r4mix "$R4,$N/sq_demos" $FULL --steps 8000

# ToolHang, upside down: two iterations of full fine-tuning
S=${SPEC[bothflip]}
col $N/thimg_cf1_none toolhang act $ACT_TH 1 20 --selflabel cf --save-images
col $N/thimg_cf1_bothflip toolhang act $ACT_TH 1 30 --selflabel cf --save-images --nuisance "$S"
thfit() { edit $E/$1 $ACT_TH --images "$2" --stride 4 --pres none --beta 0 --batch 32 $FULL --steps 6000; }
thfit th_depth_full_r1 "$N/thimg_cf1_bothflip,$N/thimg_cf1_none"
col $N/th_depth_full_r1_eval3_bothflip toolhang act $E/th_depth_full_r1 3 30 --nuisance "$S"
col $N/thimg_cf2_bothflip toolhang act $E/th_depth_full_r1 2 30 --selflabel cf --save-images --nuisance "$S" --label-weights $ACT_TH
thfit th_depth_full_r2 "$N/thimg_cf1_bothflip,$N/thimg_cf2_bothflip,$N/thimg_cf1_none"
col $N/th_depth_full_r2_eval3_bothflip toolhang act $E/th_depth_full_r2 3 30 --nuisance "$S"
col $N/th_depth_full_r2_eval3_none toolhang act $E/th_depth_full_r2 3 30

python scripts/interp_grid.py --table depth
