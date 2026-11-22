"""Retraining the head (the readout) of a frozen policy, and going deeper when that is not enough.

  ACT       closed form: weighted ridge on the head inputs towards the label chunks, change
            truncated to rank r (head_closed_form); or SGD on what the policy saw: a rank-r update
            W + B A on one site or every linear layer (LoRA), or every weight (train_act)
  DP        the FiLM conditioning layers, the only path from the observation features into the
            denoiser, with DP's own denoising loss (train_dp)
  SmolVLA   action_out_proj, the flow-matching velocity readout, with its own loss (train_vla)
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np

from .act import edit_targets, low_rank, normalize_actions, run_frames
from .stats import weighted_ridge


def save_pretrained(model, parent: str, out: str, stats: dict) -> None:
    """A LeRobot pretrained_model dir (the parent's other files copied over) plus edit.json."""
    model.save_pretrained(out)
    for f in os.listdir(parent):
        if not os.path.exists(os.path.join(out, f)):
            shutil.copy(os.path.join(parent, f), out)
    with open(os.path.join(out, 'edit.json'), 'w') as f:
        json.dump(stats, f, indent=1)


def predict_frames(act, data, batch: int = 256):
    """Normalised chunks for every frame of a replay/recorded frames tuple."""
    import torch
    with torch.no_grad():
        return torch.cat([run_frames(act, data[0], data[1], np.arange(i, min(i + batch, len(data[1]))))
                          for i in range(0, len(data[1]), batch)])


def head_inputs(act, data, batch: int = 256) -> np.ndarray:
    """ACT's head inputs (frames, 16, 512) for every frame of a frames tuple."""
    import torch
    grab = {}
    hook = act.model.decoder.register_forward_hook(lambda _m, _i, o: grab.__setitem__('d', o))
    out = []
    for i in range(0, len(data[1]), batch):
        with torch.no_grad():
            run_frames(act, data[0], data[1], np.arange(i, min(i + batch, len(data[1]))))
        out.append(grab['d'].transpose(0, 1).double().cpu().numpy())
    hook.remove()
    return np.concatenate(out)


def head_closed_form(act, eps: list, keep: np.ndarray, beta: float, rank: int, seed: int) -> dict:
    """Refit ACT's linear head on recorded decisions (collect.py --teacher / --selflabel: 'dec' =
    the head inputs, 'teacher' = the label chunk): weighted ridge towards the labels, with `keep`
    (head inputs whose output must not change) weighing beta against the decisions; the change
    truncated to rank r. Edits act in place."""
    import torch
    head = act.model.action_head
    W0, b0 = (t.detach().cpu().double().numpy() for t in (head.weight, head.bias))
    H = act.config.chunk_size
    Xo = np.concatenate([e['dec'] for e in eps]).astype(np.float64)       # (N, 16, D)
    with torch.no_grad():
        to = normalize_actions(act, np.concatenate([e['teacher'] for e in eps]).astype(np.float32))
    Yo = to.double().cpu().numpy() - (Xo @ W0.T + b0)
    go = np.concatenate([np.full(len(e['dec']), i) for i, e in enumerate(eps)])
    if len(Xo) > 20000:                         # consecutive ticks are near-duplicates
        sub = np.sort(np.random.default_rng(seed).choice(len(Xo), 20000, replace=False))
        Xo, Yo, go = Xo[sub], Yo[sub], go[sub]
    rows = W0.shape[0]
    Xo, Yo = Xo.reshape(-1, Xo.shape[-1]), Yo.reshape(-1, rows)
    X, Y = np.vstack([Xo, keep]), np.vstack([Yo, np.zeros((len(keep), rows))])
    w = np.r_[np.full(len(Xo), 1.0 / len(Xo)),
              np.full(len(keep), max(beta, 1e-6) / max(len(keep), 1))]
    groups = np.r_[np.repeat(go, H), go.max() + 1 + np.arange(len(keep)) % 5]
    dW, db, lam = weighted_ridge(X, Y, w, groups)
    if rank < rows:                             # truncate the change to rank r
        U, S, Vt = np.linalg.svd(dW, full_matrices=False)
        dW = (U[:, :rank] * S[:rank]) @ Vt[:rank]
    with torch.no_grad():
        head.weight += torch.from_numpy(dW).to(head.weight)
        head.bias += torch.from_numpy(db).to(head.bias)
    print(f'closed form: lambda {lam:g}, |dW| {np.linalg.norm(dW):.3f}', flush=True)
    return dict(params=dW.size + db.size, decisions=len(go))


def train_act(act, site: str, rank: int, edit, pres=None, parent_pres=None, *, beta: float = 1.0,
              steps: int = 1000, batch: int = 32, lr: float = 1e-3, seed: int = 0) -> int:
    """SGD on what the policy saw (a frames tuple: images, state, chunk, pad mask, episode):
    L1 on normalised chunks + beta x L1 to the parent's outputs `parent_pres` on `pres`. site
    'full' trains every weight; otherwise W + B A (rank r, B = 0 at the start) on that site or,
    'all', every linear layer. Returns the number of trained values; edits act in place."""
    import torch
    import torch.nn.utils.parametrize as P
    dev = next(act.parameters()).device
    act.requires_grad_(site == 'full')
    if site == 'full':
        params = list(act.parameters())
    else:
        params, tgts = [], edit_targets(site, act.model)
        for i, (mod, name) in enumerate(tgts):
            lr_mod = low_rank(getattr(mod, name).shape, rank, seed + i).to(dev)
            P.register_parametrization(mod, name, lr_mod)
            params += list(lr_mod.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    for step in range(steps):
        i = rng.integers(len(edit[1]), size=batch)
        m = ~torch.from_numpy(edit[3][i]).to(dev)
        loss = (run_frames(act, edit[0], edit[1], i) - normalize_actions(act, edit[2][i])).abs().mean(-1)[m].mean()
        if beta and pres is not None:
            j = rng.integers(len(pres[1]), size=batch)
            loss = loss + beta * (run_frames(act, pres[0], pres[1], j) - parent_pres[j]).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            print(f'step {step}: loss {loss.item():.4f}', flush=True)
    for mod, name in ([] if site == 'full' else tgts):
        P.remove_parametrizations(mod, name, leave_parametrized=True)
    return sum(p.numel() for p in params)


def train_dp(dp, cond: np.ndarray, traj: np.ndarray, *, steps: int = 4000, batch: int = 256,
             lr: float = 1e-4, seed: int = 0) -> tuple[int, float]:
    """Refit DP's FiLM conditioning Linear layers (cond_encoder) with its denoising loss on
    (global_cond (N, D), label trajectories abs10 (N, H, 10)); the encoder is frozen, so the
    cached global_cond is exactly what the edited policy computes. Returns (trained values, final
    loss); edits dp in place."""
    import torch
    dev = next(dp.parameters()).device
    with torch.no_grad():
        target = dp.normalizer['action'].normalize(torch.from_numpy(traj).to(dev))
    cond = torch.from_numpy(cond).to(dev)
    sched = dp.noise_scheduler
    pred_type = sched.config.prediction_type
    dp.requires_grad_(False)
    params = [p for n, p in dp.model.named_parameters() if 'cond_encoder' in n]
    for q in params:
        q.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    dp.model.train()
    for step in range(steps):
        i = torch.from_numpy(rng.integers(len(cond), size=batch)).to(dev)
        x0 = target[i]
        noise = torch.randn_like(x0)
        t = torch.randint(0, sched.config.num_train_timesteps, (len(i),), device=dev).long()
        pred = dp.model(sched.add_noise(x0, noise, t), t, global_cond=cond[i])
        loss = torch.nn.functional.mse_loss(pred, noise if pred_type == 'epsilon' else x0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            print(f'step {step}: loss {loss.item():.4f}', flush=True)
    dp.model.eval()
    return sum(q.numel() for q in params), float(loss.item())


def train_vla(policy, agent, wrist, prop, lab, *, site: str = 'out', steps: int = 2000,
              batch: int = 16, lr: float = 1e-4, seed: int = 0) -> tuple[int, float]:
    """Refit SmolVLA's action_out_proj (site 'expert': also the action expert) with its own
    flow-matching loss on what it saw and the label chunks. Returns (trained values, final loss)."""
    import torch
    vla = policy._model
    dev = next(vla.parameters()).device
    keys = list(vla.config.image_features)                   # agentview first, then the wrist
    H = vla.config.chunk_size
    vla.requires_grad_(False)
    mods = [vla.model.action_out_proj] + ([vla.model.vlm_with_expert.lm_expert] if site == 'expert' else [])
    params = [q for m in mods for q in m.parameters()]
    for q in params:
        q.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    vla.train()
    for step in range(steps):
        i = rng.integers(len(lab), size=batch)
        img = [torch.from_numpy(x[i]).to(dev).permute(0, 3, 1, 2).float() / 255 for x in (agent, wrist)]
        b = {keys[0]: img[0], keys[1]: img[1],
             'observation.state': torch.from_numpy(prop[i]).to(dev),
             'action': torch.from_numpy(lab[i][:, :H]).to(dev),
             'task': [policy.instruction] * len(i)}
        loss, _ = vla.forward(b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            print(f'step {step}: loss {loss.item():.4f}', flush=True)
    vla.eval()
    return sum(q.numel() for q in params), float(loss.item())
