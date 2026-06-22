"""Run a trained corrector to produce corrected forecasts, and persist models."""

import numpy as np
import torch

from .model import Seq2SeqCorrector


def predict(model, data, indices, channels: str) -> dict:
    """Corrected wind & gust forecasts (knots) for the given window indices.

    corrected = AROME forecast + predicted residual (de-standardized).
    Returns {'wind': (n, F), 'gust': (n, F), 'residual': (n, F, 2)}.
    """
    history_idx, future_idx = data.channel_indices(channels)
    selected = torch.tensor(np.asarray(indices))
    model.eval()
    with torch.no_grad():
        residual_std = model(
            data.history_features[selected][:, :, history_idx],
            data.future_features[selected][:, :, future_idx],
        ).numpy()
    residual = residual_std * data.residual_std + data.residual_mean   # back to knots
    return {
        "wind": data.arome_forecast_wind[indices] + residual[:, :, 0],
        "gust": data.arome_forecast_gust[indices] + residual[:, :, 1],
        "residual": residual,
    }


def save_model(model, data, channels: str, path) -> None:
    """Persist weights + the channel/scaler metadata needed to run inference later."""
    history_idx, future_idx = data.channel_indices(channels)
    torch.save({
        "state_dict": model.state_dict(),
        "channels": channels,
        "history_feature_names": data.history_feature_names,
        "future_feature_names": data.future_feature_names,
        "n_history_features": len(history_idx),
        "n_future_features": len(future_idx),
        "hidden": model.encoder.hidden_size,
        "history_hours": data.history_hours,
        "horizon_hours": data.horizon_hours,
        "scalers": {
            "history_mean": data.history_mean, "history_std": data.history_std,
            "future_mean": data.future_mean, "future_std": data.future_std,
            "residual_mean": data.residual_mean, "residual_std": data.residual_std,
        },
    }, path)


def load_model(path):
    """Reconstruct a model + its metadata dict from a checkpoint saved by save_model."""
    ckpt = torch.load(path, weights_only=False)
    model = Seq2SeqCorrector(ckpt["n_history_features"], ckpt["n_future_features"], ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
