"""Per-lead error metrics and model-vs-baseline comparison tables."""

import numpy as np
import pandas as pd


def perlead_rmse(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """RMSE per lead hour. pred/truth shape (n_windows, F) -> (F,)."""
    return np.sqrt(np.mean((pred - truth) ** 2, axis=0))


def comparison_table(data, corrected: dict, target: str = "wind") -> pd.DataFrame:
    """Per-lead RMSE for raw AROME, persistence, and each corrected model.

    `corrected` maps model name -> corrected forecast array (already restricted to
    the validation windows). `target` is 'wind' or 'gust'.
    """
    val_idx = data.val_idx
    if target == "wind":
        truth = data.observed_wind[val_idx]
        arome = data.arome_forecast_wind[val_idx]
        persistence = data.persistence_wind[val_idx]
    else:
        truth = data.observed_gust[val_idx]
        arome = data.arome_forecast_gust[val_idx]
        persistence = data.persistence_gust[val_idx]

    columns = {
        "AROME_raw": perlead_rmse(arome, truth),
        "persistence": perlead_rmse(persistence, truth),
    }
    for name, pred in corrected.items():
        columns[name] = perlead_rmse(pred, truth)

    table = pd.DataFrame(columns, index=np.arange(1, data.horizon_hours + 1))
    table.index.name = "lead_h"
    return table
