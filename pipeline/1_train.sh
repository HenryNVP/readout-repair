#!/usr/bin/env bash
# The base policies: ACT on Lift (the base arm, plus the x- and y-coverage retrains Table 2 uses
# as the 400-demo upper bound and DAgger experts), Square and ToolHang; SmolVLA fine-tuned on the
# ACT Lift arm's demos and on Square; Diffusion Policy on ToolHang (the authors released no ToolHang
# image checkpoint; Lift and Square are theirs, from 0_fetch.sh). ~100k steps per ACT, batch 8:
# about a GPU-day for the four on one consumer GPU; each SmolVLA ~10 h and DP ToolHang ~12 h on a
# 16 GB card.
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

# SmolVLA on Square: the same fine-tune on the Square demos
[ -f $VLA_SQ/instruction.txt ] || { lerobot-train --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=local/square_ph_abs6 --dataset.root=$PWD/$O/square/lerobot/square_ph_abs6 \
    --policy.device=cuda --policy.push_to_hub=false --policy.chunk_size=16 --policy.n_action_steps=16 \
    --batch_size=16 --steps=30000 --save_freq=10000 --output_dir=$PWD/$X/smolvla_square \
    --wandb.enable=false > $X/train_square.log 2>&1 \
  && echo 'put the square nut on the square peg' > $VLA_SQ/instruction.txt; }

# Diffusion Policy on ToolHang: the authors' ToolHang recipe (pipeline/dp_toolhang_84.yaml) on the
# same 84 px renders, with absolute actions derived as for ACT
A=$O/toolhang; mkdir -p $A/dp
[ -f $A/tool_hang_84_abs.hdf5 ] || python scripts/dp_abs_hdf5.py --dataset $A/tool_hang_84.hdf5 \
  --out $A/tool_hang_84_abs.hdf5 > $A/abs.log 2>&1
[ -f $DP_TH ] || python scripts/train_dp.py --config-dir=$PWD/pipeline --config-name=dp_toolhang_84 \
  hydra.run.dir=$PWD/$A/dp/run task.dataset_path=$PWD/$A/tool_hang_84_abs.hdf5 dataloader.num_workers=4 \
  > $A/dp/train.log 2>&1
echo 'train ok'
