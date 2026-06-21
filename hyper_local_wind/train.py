"""Training loop with early stopping + best-checkpoint restore."""

import numpy as np
import torch
import torch.nn as nn

from .model import Seq2SeqCorrector


def train_model(data, channels: str, cfg):
    """Train a Seq2SeqCorrector on the given channel set.

    Returns (model, info) where info = {history, best_epoch, best_val, channels}.
    The returned model has the lowest-val-loss weights loaded.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    seq_idx, fut_idx = data.channel_indices(channels)
    Xh, Xf, Y = data.Xh[:, :, seq_idx], data.Xf[:, :, fut_idx], data.Y
    tri, vai = torch.tensor(data.train_idx), torch.tensor(data.val_idx)

    model = Seq2SeqCorrector(len(seq_idx), len(fut_idx), cfg.hidden).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)
    lossf = nn.MSELoss()

    history = {"train": [], "val": []}
    best, bad, best_state, best_ep = 1e9, 0, None, -1
    for ep in range(cfg.max_epochs):
        model.train()
        perm = tri[torch.randperm(len(tri))]
        running = 0.0
        for j in range(0, len(perm), cfg.batch_size):
            b = perm[j:j + cfg.batch_size]
            opt.zero_grad()
            loss = lossf(model(Xh[b], Xf[b]), Y[b])
            loss.backward()
            opt.step()
            running += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xh[vai], Xf[vai]), Y[vai]).item()
        history["train"].append(running / len(tri))
        history["val"].append(vl)
        if vl < best - 1e-4:
            best, bad, best_ep = vl, 0, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= cfg.patience:
            break

    model.load_state_dict(best_state)
    return model, {"history": history, "best_epoch": best_ep, "best_val": best, "channels": channels}
