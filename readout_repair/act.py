"""ACT internals the probes and edits reach into (LeRobot's ACT, chunk 16, two cameras).

The decoder input is zeros, so its cross-attention queries are constant: everything the action
knows about the scene passes through one attention read of the encoder output, and the action
head is a single linear map of the decoder output (16 rows x 512).
"""
from __future__ import annotations

import numpy as np

SITES = ('mem', 'xattn', 'dec')   # encoder output, decoder cross-attention output, head input
# edit sites: name -> (module path under ACT.model, parameter)
EDIT_SITES = {
    'img_proj': ('encoder_img_feat_input_proj', 'weight'),
    'enc3_ff2': ('encoder.layers.3.linear2', 'weight'),
    'xattn_in': ('decoder.layers.0.multihead_attn', 'in_proj_weight'),
    'xattn_out': ('decoder.layers.0.multihead_attn.out_proj', 'weight'),
    'dec_ff1': ('decoder.layers.0.linear1', 'weight'),
    'dec_ff2': ('decoder.layers.0.linear2', 'weight'),
    'head': ('action_head', 'weight'),
}


class Sites:
    """Forward hooks on the three SITES: keep each output; `swap[name]` replaces it."""

    def __init__(self, act) -> None:
        self.out, self.swap, self.attn = {}, {}, None
        m = act.model
        for name, mod in (('mem', m.encoder), ('xattn', m.decoder.layers[0].multihead_attn),
                          ('dec', m.decoder)):
            mod.register_forward_hook(self._hook(name))

    def _hook(self, name):
        def h(_m, _i, out):
            t = self.swap.get(name, out[0] if isinstance(out, tuple) else out)
            self.out[name] = t
            if isinstance(out, tuple):
                self.attn = out[1]
                return t, out[1]
            return t
        return h


def forward(act, batch):
    """predict_action_chunk with gradients: (abs10 chunk (16, 10), target row, target xy), the
    target being the last open-gripper entry."""
    import torch
    b = dict(act.normalize_inputs(batch))
    b['observation.images'] = [b[k] for k in act.config.image_features]
    a = act.model(b)[0][0]
    # LeRobot's Unnormalize runs under no_grad; it is affine, so apply it by hand
    u0, u1 = (act.unnormalize_outputs({'action': torch.full_like(a, v)})['action'] for v in (0., 1.))
    chunk = a * (u1 - u0) + u0
    open_ = (chunk[:, 9] <= 0).nonzero()
    i = int(open_[-1]) if len(open_) else chunk.shape[0] - 1
    return chunk, i, chunk[i, :2]


def run_frames(act, img, state, idx):
    """Normalised chunks for frames idx of (per-camera uint8 HWC images, proprio states)."""
    import torch
    dev = next(act.parameters()).device
    keys = list(act.config.image_features)
    b = {k: torch.from_numpy(img[c][idx]).to(dev).permute(0, 3, 1, 2).float() / 255
         for c, k in enumerate(keys)}
    b['observation.state'] = torch.from_numpy(state[idx]).to(dev)
    b = dict(act.normalize_inputs(b))
    b['observation.images'] = [b[k] for k in keys]
    return act.model(b)[0]


def normalize_actions(act, chunk: np.ndarray):
    import torch
    dev = next(act.parameters()).device
    return act.normalize_targets({'action': torch.from_numpy(chunk).to(dev)})['action']


def low_rank(shape, r: int, seed: int):
    """W -> W + B A, A seeded, B zero: the edit starts as the parent."""
    import torch
    from torch import nn

    class LowRank(nn.Module):
        def __init__(self):
            super().__init__()
            out, inp = shape[0], int(np.prod(shape[1:]))
            g = torch.Generator().manual_seed(seed)
            self.A = nn.Parameter(torch.randn(r, inp, generator=g) / inp ** 0.5)
            self.B = nn.Parameter(torch.zeros(out, r))

        def forward(self, W):
            return W + (self.B @ self.A).view_as(W)
    return LowRank()


def edit_targets(site: str, model) -> list[tuple]:
    """(module, parameter name) pairs an edit touches; 'all' = every site and every encoder /
    decoder linear and attention projection."""
    if site != 'all':
        path, name = EDIT_SITES[site]
        return [(model.get_submodule(path), name)]
    out = [(model.get_submodule(p), n) for p, n in EDIT_SITES.values()]
    for _, mod in list(model.encoder.named_modules()) + list(model.decoder.named_modules()):
        for n in ('weight', 'in_proj_weight'):
            w = getattr(mod, n, None) if n in mod._parameters else None
            if w is not None and w.ndim == 2 and (mod, n) not in out:
                out.append((mod, n))
    return out
