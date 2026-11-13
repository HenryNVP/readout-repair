# Sourced by every pipeline stage: paths, perturbation specs and the rollout helper. Run stages
# from anywhere with the venv active (source ~/evh_env.sh). Every step is skipped when its output
# exists, so a stage can be rerun after an interruption.
set -euo pipefail
set -f                                       # specs and --onpolicy globs reach Python unexpanded
cd "$(dirname "${BASH_SOURCE[0]}")/.."
O=outputs/interp; N=$O/nuis; E=$O/edit; D=$O/dagger
mkdir -p $N $E $D

# base policies (1_train.sh); the Diffusion Policy checkpoints are the authors' (0_fetch.sh)
ACT_LIFT=$O/aug/train/act_lift_aug_base/checkpoints/last/pretrained_model
ACT_LIFT_X=$O/aug/train/act_lift_aug_x/checkpoints/last/pretrained_model   # 400 x-shifted demos
ACT_LIFT_Y=$O/aug/train/act_lift_aug_y/checkpoints/last/pretrained_model
ACT_SQ=$O/square/train/act_square_abs6/checkpoints/last/pretrained_model
ACT_TH=$O/toolhang/train/act_toolhang_abs6/checkpoints/last/pretrained_model
VLA=$O/vla/smolvla_lift/checkpoints/last/pretrained_model
DP_LIFT=checkpoints/dp_lift_ph_image_cnn.ckpt
DP_SQ=checkpoints/dp_square_ph_image_cnn.ckpt

# the paper's perturbations (scripts/interp_nuisance.py), each applied to both cameras
declare -A SPEC=(
  [bothcam]='cam=0,0,0,30;wroll=90'                          # turn + roll
  [darkredwroll]='table=0.75,0.2,0.15;light=0.3;wroll=90'    # red, dim + roll
  [bothflip]='roll=180;wroll=180'                            # upside down
  [bothzoom]='fov=25;wfov=30'                                # zoomed
  [faint]='objalpha=0.1'                                     # faint object
  [invisible]='objalpha=0'                                   # negative control
)
OFFSETS='-0.07,0 -0.05,0 0.05,0 0.07,0 0,-0.07 0,-0.05 0,0.05 0,0.07'   # Table 1, beyond the demos' ±3 cm

# col OUT TASK BACKEND WEIGHTS SEED EPISODES [interp_collect args...]: one recorded rollout set
col() {
  local out=$1 task=$2 backend=$3 w=$4 seed=$5 n=$6
  shift 6
  [ -f $out.done ] && return
  local deadline=20; [ $task = toolhang ] && deadline=45
  python scripts/interp_collect.py --task $task --backend $backend --weights $w --seed $seed \
    --episodes $n --deadline-s $deadline --out-dir $out "$@" > $out.log 2>&1 && touch $out.done
}

# head retrain of an ACT policy: edit OUT PARENT [interp_edit args...]
edit() {
  local out=$1 parent=$2
  shift 2
  [ -f $out/edit.json ] || python scripts/interp_edit.py --parent $parent --out $out "$@" > $out.log 2>&1
}

# The head-retrain protocol for one policy, task and perturbation: on-policy episodes on training
# scenes (seed 1) labelled with the policy's own chunk on the same state rendered without the
# perturbation (--selflabel cf), the head retrained on them plus unperturbed ones, then fresh
# scenes (seed 3): base and retrained under the perturbation, retrained unperturbed.
act_protocol() {   # task weights prefix n_unperturbed n_label n_eval tag
  local task=$1 P=$2 X=$3 tag=$7 S=${SPEC[$7]}
  col $N/${X}cf1_none $task act $P 1 $4 --selflabel cf
  col $N/${X}cf1_$tag $task act $P 1 $5 --selflabel cf --nuisance "$S"
  local pres=none; [ $task = lift ] && pres=lift      # Lift also keeps the base's demo states
  edit $E/${X}rep_$tag $P --onpolicy "$N/${X}cf1_$tag,$N/${X}cf1_none" --rank 10 --beta 3 --pres $pres
  col $N/${X}eval3_base_$tag $task act $P 3 $6 --nuisance "$S"
  col $N/${X}eval3_rep_$tag $task act $E/${X}rep_$tag 3 $6 --nuisance "$S"
  col $N/${X}eval3_rep_${tag}_none $task act $E/${X}rep_$tag 3 $6
}
dp_protocol() {   # task ckpt prefix tag
  local task=$1 P=$2 X=$3 tag=$4 S=${SPEC[$4]}
  col $N/${X}cf1_none $task dp $P 1 30 --selflabel cf --label-every 4
  col $N/${X}cf1_$tag $task dp $P 1 40 --selflabel cf --label-every 4 --nuisance "$S"
  [ -f $E/${X}rep_$tag.ckpt ] || python scripts/interp_dp_repair.py --parent $P --out $E/${X}rep_$tag.ckpt \
    --data "$N/${X}cf1_$tag,$N/${X}cf1_none" > $E/${X}rep_$tag.log 2>&1
  col $N/${X}eval3_base_$tag $task dp $P 3 100 --nuisance "$S"
  col $N/${X}eval3_rep_$tag $task dp $E/${X}rep_$tag.ckpt 3 100 --nuisance "$S"
  col $N/${X}eval3_rep_${tag}_none $task dp $E/${X}rep_$tag.ckpt 3 100
}
vla_protocol() {   # tag
  local tag=$1 S=${SPEC[$1]}
  col $N/vla_cf1_none lift smolvla $VLA 1 20 --selflabel cf --label-every 4 --save-images
  col $N/vla_cf1_$tag lift smolvla $VLA 1 30 --selflabel cf --label-every 4 --save-images --nuisance "$S"
  [ -f $E/vla_rep_$tag/edit.json ] || python scripts/interp_vla_repair.py --parent $VLA --out $E/vla_rep_$tag \
    --data "$N/vla_cf1_$tag,$N/vla_cf1_none" > $E/vla_rep_$tag.log 2>&1
  col $N/vla_eval3_base_$tag lift smolvla $VLA 3 50 --nuisance "$S"
  col $N/vla_eval3_rep_$tag lift smolvla $E/vla_rep_$tag 3 50 --nuisance "$S"
  col $N/vla_eval3_rep_${tag}_none lift smolvla $E/vla_rep_$tag 3 50
}
