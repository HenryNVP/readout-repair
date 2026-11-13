#!/usr/bin/env bash
# Inputs: the robomimic proficient-human demos, the Diffusion Policy authors' Lift and Square
# image checkpoints, and two pinned source checkouts (DP model code; robomimic 0.5 to render
# ToolHang from its recorded states). ~12 GB. Checksums are those of the files the paper used.
source "$(dirname "$0")/common.sh"

get() {   # url path md5
  [ -f $2 ] || { mkdir -p $(dirname $2); curl -fL --retry 5 --retry-all-errors -C - -o $2.part "$1" && mv $2.part $2; }
  echo "$3  $2" | md5sum -c --quiet || { echo "checksum mismatch: $2" >&2; exit 1; }
}
RM=http://downloads.cs.stanford.edu/downloads/rt_benchmark
get $RM/lift/ph/image.hdf5 data/robomimic/lift/ph/image.hdf5 22733ed6b47df994d3d06076b3bbf7fb
get $RM/square/ph/image.hdf5 data/robomimic/square/ph/image.hdf5 7d798b573b1e288f96363c30d2a957c0
get $RM/tool_hang/ph/demo.hdf5 data/robomimic/tool_hang/ph/demo.hdf5 4a7bacf3cd6bf09b4926102c95891bc5
DPX=https://diffusion-policy.cs.columbia.edu/data/experiments/image
get "$DPX/lift_ph/diffusion_policy_cnn/train_0/checkpoints/epoch%3D0300-test_mean_score%3D1.000.ckpt" \
  $DP_LIFT ce05c1fd7573ee9c8af5ee15ae0b730a
get "$DPX/square_ph/diffusion_policy_cnn/train_0/checkpoints/epoch%3D1250-test_mean_score%3D1.000.ckpt" \
  $DP_SQ 7cf6f2e43818622b2a186c9b7d5e45cd

clone() {   # url dir commit
  [ -d $2/.git ] || git clone -q $1 $2
  git -C $2 checkout -q $3
}
clone https://github.com/real-stanford/diffusion_policy.git external/diffusion_policy 5ba07ac
clone https://github.com/ARISE-Initiative/robomimic.git external/robomimic d309eae
echo 'fetch ok'
