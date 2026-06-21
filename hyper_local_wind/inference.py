"""Run a trained corrector to produce corrected forecasts, and persist models."""

import numpy as np
import torch

from .model import Seq2SeqCorrector


def predict(model, data, indices, channels: str) -> dict:
    """Corrected wind & gust forecasts (knots) for the given window indices.

    corrected = AROME forecast + predicted residual (de-standardized).
    Returns {'wind': (n, F), 'gust': (n, F), 'residual': (n, F, 2)}.
    """
    seq_idx, fut_idx = data.channel_indices(channels)
    sel = torch.tensor(np.asarray(indices))
    model.eval()
    with torch.no_grad():
        pred_std = model(data.Xh[sel][:, :, seq_idx], data.Xf[sel][:, :, fut_idx]).numpy()
    residual = pred_std * data.y_std + data.y_mean        # back to knots
    return {
        "wind": data.Aw[indices] + residual[:, :, 0],
        "gust": data.Ag[indices] + residual[:, :, 1],
        "residual": residual,
    }


def save_model(model, data, channels: str, path) -> None:
    """Persist weights + the channel/scaler metadata needed to run inference later."""
    seq_idx, fut_idx = data.channel_indices(channels)
    torch.save({
        "state_dict": model.state_dict(),
        "channels": channels,
        "seq_names": data.seq_names,
        "fut_names": data.fut_names,
        "n_seq": len(seq_idx),
        "n_fut": len(fut_idx),
        "hidden": model.encoder.hidden_size,
        "H": data.H,
        "F": data.F,
        "scalers": {
            "seq_mean": data.seq_mean, "seq_std": data.seq_std,
            "fut_mean": data.fut_mean, "fut_std": data.fut_std,
            "y_mean": data.y_mean, "y_std": data.y_std,
        },
    }, path)


def load_model(path):
    """Reconstruct a model + its metadata dict from a checkpoint saved by save_model."""
    ckpt = torch.load(path, weights_only=False)
    model = Seq2SeqCorrector(ckpt["n_seq"], ckpt["n_fut"], ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
