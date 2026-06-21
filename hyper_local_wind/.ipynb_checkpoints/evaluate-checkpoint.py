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
    va = data.val_idx
    if target == "wind":
        truth, arome, pers = data.OW[va], data.Aw[va], data.Pw[va]
    else:
        truth, arome, pers = data.OG[va], data.Ag[va], data.Pg[va]

    cols = {
        "AROME_raw": perlead_rmse(arome, truth),
        "persistence": perlead_rmse(pers, truth),
    }
    for name, pred in corrected.items():
        cols[name] = perlead_rmse(pred, truth)

    table = pd.DataFrame(cols, index=np.arange(1, data.F + 1))
    table.index.name = "lead_h"
    return table
