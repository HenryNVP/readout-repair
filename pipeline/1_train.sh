#!/usr/bin/env bash
# The base policies: ACT on Lift (the base arm, plus the x- and y-coverage retrains Table 1 uses
# as the 400-demo upper bound and DAgger experts), Square and ToolHang; SmolVLA fine-tuned on the
# ACT Lift arm's demos. ~100k steps each, batch 8: about a GPU-day for the four ACTs together on
# one consumer GPU, and SmolVLA ~10 h on a 16 GB card. The Diffusion Policy checkpoints are not
# retrained (0_fetch.sh downloads the authors').
source "$(dirname "$0")/common.sh"

act() {   # out dataset_root steps
  [ -d $1 ] || lerobot-train --dataset.repo_id=local/$(basename $2) --dataset.root=$PWD/$2 \
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
    --policy.chunk_size=16 --policy.n_action_steps=16 --policy.temporal_ensemble_coeff=null \
    --batch_size=8 --steps=$3 --save_freq=25000 --output_dir=$PWD/$1 > $1.log 2>&1
  python scripts/stamp_act_checkpoint.py $1/checkpoints/last/pretrained_model --dataset-root $2
}

# Lift: the 200 demos re-rendered by the eval renderer (base), and the same with 200 more demos
# replayed with the cube translated along one axis (x, y; the coverage arms)
A=$O/aug; mkdir -p $A/train
for arm in base x y; do
  if [ $arm = base ]; then extra="--axis x --copies 0"; else extra="--axis $arm"; fi
  [ -f $A/lift_$arm.hdf5 ] || python scripts/augment_lift.py $extra --out $A/lift_$arm.hdf5 > $A/gen_$arm.log 2>&1
  [ -d $A/lerobot/lift_$arm ] || python scripts/to_lerobot.py --dataset $A/lift_$arm.hdf5 \
    --out $A/lerobot/lift_$arm --actions abs --check-episodes 0 > $A/conv_$arm.log 2>&1
  act $A/train/act_lift_aug_$arm $A/lerobot/lift_$arm 100000
done

A=$O/square; mkdir -p $A/train
[ -d $A/lerobot/square_ph_abs6 ] || python scripts/to_lerobot.py \
  --dataset data/robomimic/square/ph/image.hdf5 --out $A/lerobot/square_ph_abs6 \
  --actions abs-derived --check-episodes 2 > $A/conv.log 2>&1
act $A/train/act_square_abs6 $A/lerobot/square_ph_abs6 100000

# ToolHang ships states only: render 84 px agentview + wrist images first
A=$O/toolhang; mkdir -p $A/train
[ -f $A/tool_hang_84.hdf5 ] || python scripts/render_toolhang.py --dataset data/robomimic/tool_hang/ph/demo.hdf5 \
  --out $A/tool_hang_84.hdf5 > $A/regen.log 2>&1
[ -d $A/lerobot/toolhang_ph_abs6 ] || python scripts/to_lerobot.py --dataset $A/tool_hang_84.hdf5 \
  --out $A/lerobot/toolhang_ph_abs6 --actions abs-derived --check-episodes 2 > $A/conv.log 2>&1
act $A/train/act_toolhang_abs6 $A/lerobot/toolhang_ph_abs6 100000

# SmolVLA: lerobot/smolvla_base fine-tuned on the ACT Lift base arm's demos, chunk 16 like ACT
X=$O/vla; mkdir -p $X
[ -f $VLA/instruction.txt ] || { lerobot-train --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=local/lift_base --dataset.root=$PWD/$O/aug/lerobot/lift_base \
    --policy.device=cuda --policy.push_to_hub=false --policy.chunk_size=16 --policy.n_action_steps=16 \
    --batch_size=16 --steps=30000 --save_freq=5000 --output_dir=$PWD/$X/smolvla_lift \
    --wandb.enable=false > $X/train.log 2>&1 \
  && echo 'lift the cube' > $VLA/instruction.txt; }
echo 'train ok'
