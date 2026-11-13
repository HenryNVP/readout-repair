"""Policy backends behind one interface: obs dict -> action chunk [chunk_size, 7].

  * ACTBackend      -- LeRobot ACT checkpoint (one forward per chunk). A 10-dim head is the
                       abs-action [pos, rot_6d, gripper] layout and is converted to 7-dim.
  * SmolVLABackend  -- LeRobot SmolVLA checkpoint: ACT's observation/action contract plus the
                       language instruction.
  * DiffusionPolicyRepoBackend (dp_repo_policy.py) -- real-stanford/diffusion_policy robomimic
                       image checkpoints.

Observation contract: a dict whose values may carry a history axis (the last `n_obs_steps`
control ticks, oldest first):

  obs['agentview']  uint8 [To, H, W, 3] or [H, W, 3]
  obs['wrist']      uint8 [To, H, W, 3]           (optional; backends declare needs_wrist)
  obs['proprio']    float [To, D] or [D]          ([eef_pos(3), eef_quat(4), gripper_qpos(2)])

Actions are 7-dim [pos(3), axis-angle(3), gripper], absolute end-effector targets for the
abs-action checkpoints (`absolute_actions`), deltas otherwise.
"""
from __future__ import annotations

import importlib
import logging
import os
from abc import ABC, abstractmethod

import numpy as np

from rotation import abs10_to_abs7

logger = logging.getLogger(__name__)


def _import_act_policy():
    """Load ACTPolicy across LeRobot 0.3.x package layouts."""
    tried: list[str] = []
    for modpath in (
        'lerobot.policies.act.modeling_act',
        'lerobot.common.policies.act.modeling_act',
    ):
        try:
            return importlib.import_module(modpath).ACTPolicy
        except Exception as exc:
            tried.append(f'  {modpath}: {type(exc).__name__}: {exc}')
    raise ImportError(
        'Could not import LeRobot ACTPolicy. Tried:\n'
        + '\n'.join(tried)
        + '\nInstall lerobot==0.3.3 (Python 3.10+).'
    )


def _resolve_device(requested: str) -> str:
    import torch

    if requested.startswith('cuda') and not torch.cuda.is_available():
        logger.warning('CUDA requested but unavailable; using CPU for policy inference.')
        return 'cpu'
    return requested


class ChunkPolicy(ABC):
    """obs dict (see module docstring) -> action chunk [chunk_size, action_dim]."""

    action_dim: int
    chunk_size: int
    denoise_steps: int
    n_obs_steps: int = 1            # history depth the rollout must keep
    needs_wrist: bool = False       # whether obs['wrist'] is required
    absolute_actions: bool = False  # actions are absolute EE pose targets, not deltas

    @abstractmethod
    def predict(self, obs: dict) -> np.ndarray:
        ...


POLICY_SIDECAR = 'evh_policy.json'


def sidecar_absolute(ckpt_dir: str) -> bool | None:
    """Read `evh_policy.json` next to a checkpoint: `{"absolute_actions": true}`, or None.

    A 7-dim ACT checkpoint cannot tell absolute from delta actions (a 10-dim head is always
    absolute), so the mode is stamped beside the weights at training time
    (stamp_act_checkpoint.py) and read back here. Missing file -> None.
    """
    import json
    import os

    path = os.path.join(ckpt_dir, POLICY_SIDECAR) if os.path.isdir(ckpt_dir) else ''
    if not path or not os.path.isfile(path):
        return None
    try:
        value = json.loads(open(path).read()).get('absolute_actions')
    except Exception as exc:                       # a malformed stamp must not be read as False
        raise ValueError(f'could not read {path}: {exc}') from exc
    return None if value is None else bool(value)


def resolve_absolute(override: bool | None, stamped: bool | None, source: str) -> bool:
    """Settle a backend's action convention: explicit override, else the stamp, else delta."""
    if override is not None:
        return bool(override)
    if stamped is not None:
        return bool(stamped)
    logger.warning(
        '%s carries no action-convention stamp — assuming DELTA actions. If it was trained on '
        'absolute actions, stamp it (scripts/stamp_act_checkpoint.py).', source,
    )
    return False


def newest(obs_value: np.ndarray) -> np.ndarray:
    """Latest entry of a possibly history-stacked observation value."""
    arr = np.asarray(obs_value)
    return arr[-1] if arr.ndim in (2, 4) else arr


class ACTBackend(ChunkPolicy):
    """LeRobot ACT checkpoint — one transformer forward per chunk (no denoise loop)."""

    def __init__(self, ckpt_path: str, device: str = 'cuda',
                 absolute: bool | None = None) -> None:
        self.ckpt_path = ckpt_path
        self.device = device
        self._absolute_override = absolute
        if absolute is not None:                    # applies even with no checkpoint to load
            self.absolute_actions = bool(absolute)
        self.denoise_steps = 1
        self.action_dim = 7
        self.chunk_size = 16
        self.n_obs_steps = 1
        self._model = None
        self._torch_device = 'cpu'
        self._image_keys: list[str] = []
        self._image_shapes: dict[str, tuple[int, ...]] = {}
        self._state_key = 'observation.state'
        self._state_dim = 7
        self._load()

    def _load(self) -> None:
        if not self.ckpt_path:
            self._model = None
            return

        ACTPolicy = self._import_policy()
        self._torch_device = _resolve_device(self.device)
        logger.info('Loading ACT policy from %s on %s', self.ckpt_path, self._torch_device)
        self._model = ACTPolicy.from_pretrained(self.ckpt_path)
        self._model.to(self._torch_device).eval()

        cfg = self._model.config
        self.chunk_size = int(cfg.chunk_size)
        self.n_obs_steps = int(getattr(cfg, 'n_obs_steps', 1))
        if cfg.action_feature and cfg.action_feature.shape:
            raw_action_dim = int(cfg.action_feature.shape[0])
            # A 10-dim head is [pos(3), rot_6d(6), gripper] — the abs-action layout, and the same
            # tell the DP backend uses. Absolute rotation is trained as 6D because axis-angle is
            # discontinuous exactly where these targets live (see rotation.py), so a 10-dim ACT
            # announces its own mode and needs no stamp.
            self._rot6d = raw_action_dim == 10
            self.action_dim = 7 if self._rot6d else raw_action_dim

        self._image_keys = list(cfg.image_features.keys()) if cfg.image_features else []
        self._image_shapes = {
            key: tuple(cfg.input_features[key].shape)
            for key in self._image_keys
        }

        if self._state_key not in cfg.input_features:
            for key, feat in cfg.input_features.items():
                feat_type = getattr(feat, 'type', None)
                if feat_type is not None and str(feat_type).endswith('STATE'):
                    self._state_key = key
                    break
                if 'state' in key:
                    self._state_key = key
                    break

        state_shape = cfg.input_features[self._state_key].shape
        self._state_dim = int(state_shape[0]) if state_shape else self._state_dim
        self.needs_wrist = len(self._image_keys) > 1

        stamped = True if self._rot6d else sidecar_absolute(self.ckpt_path)
        self.absolute_actions = resolve_absolute(
            self._absolute_override, stamped, self.ckpt_path)

        logger.info(
            'Loaded ACT policy: chunk_size=%d action_dim=%d state_dim=%d image_keys=%s '
            'absolute=%s',
            self.chunk_size,
            self.action_dim,
            self._state_dim,
            self._image_keys or ['<none>'],
            self.absolute_actions,
        )

    def _import_policy(self):
        return _import_act_policy()

    def _obs_to_batch(self, obs: dict) -> dict:
        import torch
        import torch.nn.functional as F

        batch: dict[str, torch.Tensor] = {}
        for key in self._image_keys:
            if key == self._image_keys[0]:
                image = np.asarray(newest(obs['agentview']), dtype=np.uint8)
            elif len(self._image_keys) > 1 and key == self._image_keys[1]:
                image = np.asarray(newest(obs.get('wrist', obs['agentview'])), dtype=np.uint8)
            else:
                image = np.asarray(newest(obs['agentview']), dtype=np.uint8)

            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f'expected HWC uint8 image, got shape {image.shape}')
            img = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            c, h, w = self._image_shapes[key]
            if (img.shape[0], img.shape[1], img.shape[2]) != (c, h, w):
                img = F.interpolate(
                    img.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False,
                )[0]
            batch[key] = img.unsqueeze(0).to(self._torch_device)

        state = newest(obs['proprio'])
        state_vec = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_vec.size < self._state_dim:
            state_vec = np.pad(state_vec, (0, self._state_dim - state_vec.size))
        elif state_vec.size > self._state_dim:
            state_vec = state_vec[:self._state_dim]
        batch[self._state_key] = (
            torch.from_numpy(state_vec).unsqueeze(0).to(self._torch_device)
        )
        return batch

    def predict(self, obs: dict) -> np.ndarray:
        if self._model is None:
            return np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)

        import torch

        batch = self._obs_to_batch(obs)
        with torch.inference_mode():
            chunk = self._model.predict_action_chunk(batch)[0].cpu().numpy()
        n = min(self.chunk_size, chunk.shape[0])
        chunk = np.asarray(chunk[:n], dtype=np.float32)
        return abs10_to_abs7(chunk) if self._rot6d else chunk


class SmolVLABackend(ACTBackend):
    """LeRobot SmolVLA checkpoint: the ACT backend's observation and action contract, plus the
    language instruction it was fine-tuned with (instruction.txt beside the weights, else the
    Lift sentence robomimic_to_lerobot writes)."""

    def _import_policy(self):
        return importlib.import_module('lerobot.policies.smolvla.modeling_smolvla').SmolVLAPolicy

    def _load(self) -> None:
        super()._load()
        path = os.path.join(self.ckpt_path, 'instruction.txt') if self.ckpt_path else ''
        self.instruction = (open(path).read().strip() if path and os.path.exists(path)
                            else 'lift the cube')

    def _obs_to_batch(self, obs: dict) -> dict:
        batch = super()._obs_to_batch(obs)
        batch['task'] = [self.instruction]
        return batch


def make_policy(backend: str, weights_path: str, denoise_steps: int = 16,
                absolute: bool | None = None) -> ChunkPolicy:
    """`absolute` overrides the action convention a backend derives for itself (None = derive)."""
    backend = backend.lower()
    if backend in ('dp', 'diffusion_policy'):
        from dp_repo_policy import DiffusionPolicyRepoBackend
        return DiffusionPolicyRepoBackend(weights_path, denoise_steps=denoise_steps)
    if backend in ('act', 'act_lerobot'):
        return ACTBackend(weights_path, absolute=absolute)
    if backend == 'smolvla':
        return SmolVLABackend(weights_path, absolute=absolute)
    raise ValueError(f'unknown backend: {backend!r}')
