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

    history_idx, future_idx = data.channel_indices(channels)
    history_inputs = data.history_features[:, :, history_idx]
    future_inputs = data.future_features[:, :, future_idx]
    target = data.residual_target
    train_idx = torch.tensor(data.train_idx)
    val_idx = torch.tensor(data.val_idx)

    model = Seq2SeqCorrector(len(history_idx), len(future_idx), cfg.hidden).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    loss_history = {"train": [], "val": []}
    best_val, epochs_without_improvement, best_state, best_epoch = 1e9, 0, None, -1
    for epoch in range(cfg.max_epochs):
        model.train()
        shuffled = train_idx[torch.randperm(len(train_idx))]
        running_loss = 0.0
        for start in range(0, len(shuffled), cfg.batch_size):
            batch = shuffled[start:start + cfg.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(history_inputs[batch], future_inputs[batch]), target[batch])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(history_inputs[val_idx], future_inputs[val_idx]), target[val_idx]).item()
        loss_history["train"].append(running_loss / len(train_idx))
        loss_history["val"].append(val_loss)
        if val_loss < best_val - 1e-4:
            best_val, epochs_without_improvement, best_epoch = val_loss, 0, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= cfg.patience:
            break

    model.load_state_dict(best_state)
    return model, {"history": loss_history, "best_epoch": best_epoch, "best_val": best_val, "channels": channels}
