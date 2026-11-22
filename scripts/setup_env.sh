#!/usr/bin/env bash
# Build the Python environment the experiments run in: a plain Python 3.10 venv with
# requirements.txt, three --no-deps installs (see requirements.txt for why), lerobot's own
# requirements minus pynput plus its SmolVLA extras, and a render smoke test that picks the MuJoCo
# GL backend (EGL on a GPU, else OSMesa on the CPU). Writes an env file that run scripts source.
#
#   bash scripts/setup_env.sh            # from the repo root
#
# EVH_VENV picks the venv (default .venv), EVH_ENV the env file (default ~/evh_env.sh, which the
# run scripts source), PYTHON the interpreter (default python3.10). EVH_APT=1 also apt-installs
# the GL libraries (a bare container; a desktop already has them). Idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -n "${EVH_APT:-}" ]; then
  SUDO=$([ "$(id -u)" = 0 ] || echo sudo)
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq --no-install-recommends libgl1 libegl1 libglib2.0-0 libosmesa6 \
      python3-venv > /dev/null
fi
V=$(realpath -m "${EVH_VENV:-.venv}")
ENVF=${EVH_ENV:-$HOME/evh_env.sh}
[ -x $V/bin/python ] || ${PYTHON:-python3.10} -m venv $V
PIP="$V/bin/python -m pip install -q"
$PIP -U pip
# cu128 first: PyPI's default 2.7.1 wheel is cu126, which has no kernels for Blackwell (sm_120)
$PIP torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
$PIP -r requirements.txt
$PIP --no-deps robosuite==1.4.1 lerobot==0.3.3 robomimic==0.2.0
$V/bin/python - > $V/lerobot_reqs.txt << 'PY'
import importlib.metadata as m, re
for r in m.requires('lerobot'):
    name = re.split(r'[<>=!;\[ ]', r, maxsplit=1)[0]
    extra = re.search(r'extra == "([^"]+)"', r)
    if name in ('pynput', 'cmake', 'lerobot') or (extra and extra[1] not in ('smolvla', 'transformers-dep')):
        continue
    print(r.split(';')[0].strip() if extra else r)
PY
# -c keeps lerobot's requirements from moving the pins above (torch, numpy, ...)
$PIP -r $V/lerobot_reqs.txt -c requirements.txt
$PIP --no-deps -e .                # readout_repair, the library the scripts import

cat > $ENVF << EOF
source $V/bin/activate
export TORCH_HOME=\$HOME/.cache/torch OMP_NUM_THREADS=4
EOF
source $ENVF
python -c "import torch; print('torch', torch.__version__, 'cuda' if torch.cuda.is_available() else 'CPU only')"
python -c "import torchvision.models as m; m.resnet18(weights='IMAGENET1K_V1')"
python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy"

for gl in egl osmesa; do      # render one Lift frame through the same env the evals use
  if MUJOCO_GL=$gl PYOPENGL_PLATFORM=$gl python -c "
from readout_repair.sim import TASKS, make_env
env, obs = make_env(TASKS['lift']); assert obs['agentview_image'].std() > 1; env.close()"; then
    echo "export MUJOCO_GL=$gl PYOPENGL_PLATFORM=$gl" >> $ENVF
    echo "setup ok: MUJOCO_GL=$gl; activate with: source $ENVF"
    exit 0
  fi
done
echo 'setup failed: a Lift frame rendered with neither EGL nor OSMesa' >&2
exit 1
