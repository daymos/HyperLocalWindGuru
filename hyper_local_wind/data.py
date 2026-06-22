"""
Data pipeline: decrypt -> quality-control -> feature engineering -> windowing.

The merged dataset holds, per hour: observed wind/gust/dir, plus the AROME / UKV /
ICON-D2 forecasts. We build sliding (history -> horizon) windows and the residual
target (observed - AROME) that the model learns to predict.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch

from .crypto import read_encrypted_parquet

# --- feature column groups -------------------------------------------------

# per history hour: observed + AROME + time
SEQ_BASE = [
    "wind_avg_kt", "wind_max_kt", "obs_dir_sin", "obs_dir_cos",
    "arome_wind_10m_kt", "arome_gust_10m_kt", "fc_dir_sin", "fc_dir_cos",
    "arome_temp_2m_c", "hour_sin", "hour_cos",
]
# per future hour: AROME forecast + time (observed at t+ is unknown at forecast time)
FUT_BASE = [
    "arome_wind_10m_kt", "arome_gust_10m_kt", "fc_dir_sin", "fc_dir_cos",
    "arome_temp_2m_c", "hour_sin", "hour_cos",
]
# extra NWP models (UKV + ICON-D2) — added to both history and future hours
EXTRA = [
    "ukv_wind_10m_kt", "ukv_gust_10m_kt", "ukv_dir_sin", "ukv_dir_cos",
    "icond2_wind_10m_kt", "icond2_gust_10m_kt", "icond2_dir_sin", "icond2_dir_cos",
]

# columns that must be present (drives dropna)
REQUIRED = [
    "wind_avg_kt", "wind_max_kt", "wind_dir_deg",
    "arome_wind_10m_kt", "arome_gust_10m_kt", "arome_wind_dir_deg", "arome_temp_2m_c",
    "ukv_wind_10m_kt", "ukv_gust_10m_kt", "ukv_wind_dir_deg",
    "icond2_wind_10m_kt", "icond2_gust_10m_kt", "icond2_wind_dir_deg",
]

# selectable input-channel sets (the "lead_frac" channel is appended during windowing)
CHANNELS = {
    "arome":      {"seq": SEQ_BASE,         "fut": FUT_BASE + ["lead_frac"]},
    "multimodel": {"seq": SEQ_BASE + EXTRA, "fut": FUT_BASE + EXTRA + ["lead_frac"]},
}


# --- load / clean / engineer ----------------------------------------------

def quality_control(df: pd.DataFrame):
    """Drop rows missing required fields, and dead-sensor rows (0 avg AND 0 gust)."""
    df = df.dropna(subset=REQUIRED).reset_index(drop=True)
    dead_sensor = (df["wind_avg_kt"] == 0) & (df["wind_max_kt"] == 0)
    return df[~dead_sensor].reset_index(drop=True), int(dead_sensor.sum())


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic (sin/cos) encodings for hour-of-day and every wind direction."""
    df = df.copy()
    hour = df["datetime"].dt.hour.values
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    for direction_col, sin_col, cos_col in [
        ("wind_dir_deg", "obs_dir_sin", "obs_dir_cos"),
        ("arome_wind_dir_deg", "fc_dir_sin", "fc_dir_cos"),
        ("ukv_wind_dir_deg", "ukv_dir_sin", "ukv_dir_cos"),
        ("icond2_wind_dir_deg", "icond2_dir_sin", "icond2_dir_cos"),
    ]:
        radians = np.deg2rad(df[direction_col].astype(float).values)
        df[sin_col], df[cos_col] = np.sin(radians), np.cos(radians)
    return df


def prepare(enc_path, passphrase: str):
    """Decrypt the merged archive, quality-control it, and engineer features.

    Returns (dataframe, n_dead_rows_dropped).
    """
    df = read_encrypted_parquet(enc_path, passphrase).sort_values("datetime").reset_index(drop=True)
    df, n_dead = quality_control(df)
    df = engineer_features(df)
    return df, n_dead


# --- windowing -------------------------------------------------------------

@dataclass
class WindowedData:
    """Standardized sliding-window tensors plus per-window baselines and scalers."""
    history_features: torch.Tensor       # (N, H, n_history_feat) standardized
    future_features: torch.Tensor        # (N, F, n_future_feat) standardized (incl. lead)
    residual_target: torch.Tensor        # (N, F, 2) standardized residual [wind, gust]
    history_feature_names: List[str]
    future_feature_names: List[str]
    train_idx: np.ndarray
    val_idx: np.ndarray
    arome_forecast_wind: np.ndarray      # (N, F)
    arome_forecast_gust: np.ndarray      # (N, F)
    observed_wind: np.ndarray            # (N, F) ground truth
    observed_gust: np.ndarray            # (N, F) ground truth
    persistence_wind: np.ndarray         # (N, F) last observed value held flat
    persistence_gust: np.ndarray         # (N, F)
    window_start_times: np.ndarray       # (N,) timestamp of the first forecast hour
    history_mean: np.ndarray             # input scalers (fit on train)
    history_std: np.ndarray
    future_mean: np.ndarray
    future_std: np.ndarray
    residual_mean: np.ndarray            # target scalers
    residual_std: np.ndarray
    history_hours: int
    horizon_hours: int

    def channel_indices(self, which: str):
        """Column indices into history/future features for a named channel set."""
        channels = CHANNELS[which]
        history_idx = torch.tensor([self.history_feature_names.index(c) for c in channels["seq"]])
        future_idx = torch.tensor([self.future_feature_names.index(c) for c in channels["fut"]])
        return history_idx, future_idx


def build_windows(df: pd.DataFrame, cfg) -> WindowedData:
    """Slide (history + horizon) windows within contiguous hourly runs, split
    chronologically, and standardize on training statistics."""
    history_hours, horizon_hours = cfg.history_hours, cfg.horizon_hours
    history_cols, future_cols = SEQ_BASE + EXTRA, FUT_BASE + EXTRA

    is_gap = np.r_[True, df["datetime"].diff().dt.total_seconds().values[1:] != 3600]
    run_id = np.cumsum(is_gap)

    history_matrix = df[history_cols].values.astype(np.float32)
    future_matrix = df[future_cols].values.astype(np.float32)
    obs_wind_all = df["wind_avg_kt"].values.astype(np.float32)
    obs_gust_all = df["wind_max_kt"].values.astype(np.float32)
    arome_wind_all = df["arome_wind_10m_kt"].values.astype(np.float32)
    arome_gust_all = df["arome_gust_10m_kt"].values.astype(np.float32)
    timestamps = df["datetime"].values
    lead_fraction = ((np.arange(horizon_hours) + 1) / horizon_hours).astype(np.float32)

    history_win, future_win, residual_win = [], [], []
    arome_wind, arome_gust, obs_wind, obs_gust, pers_wind, pers_gust, start_time = ([] for _ in range(7))
    for _, idx in pd.Series(np.arange(len(df))).groupby(run_id):
        idx = idx.values
        for i in range(len(idx) - (history_hours + horizon_hours) + 1):
            past = idx[i:i + history_hours]
            future = idx[i + history_hours:i + history_hours + horizon_hours]
            history_win.append(history_matrix[past])
            future_win.append(np.c_[future_matrix[future], lead_fraction])
            residual_win.append(np.stack(
                [obs_wind_all[future] - arome_wind_all[future],
                 obs_gust_all[future] - arome_gust_all[future]], 1))
            arome_wind.append(arome_wind_all[future]); arome_gust.append(arome_gust_all[future])
            obs_wind.append(obs_wind_all[future]); obs_gust.append(obs_gust_all[future])
            pers_wind.append(np.full(horizon_hours, obs_wind_all[past[-1]], np.float32))
            pers_gust.append(np.full(horizon_hours, obs_gust_all[past[-1]], np.float32))
            start_time.append(timestamps[future[0]])

    history_win = np.array(history_win)
    future_win = np.array(future_win)
    residual_win = np.array(residual_win, np.float32)
    arome_wind, arome_gust = np.array(arome_wind), np.array(arome_gust)
    obs_wind, obs_gust = np.array(obs_wind), np.array(obs_gust)
    pers_wind, pers_gust = np.array(pers_wind), np.array(pers_gust)
    start_time = np.array(start_time)

    order = np.argsort(start_time)
    cut = int(len(order) * cfg.train_frac)
    train_idx, val_idx = order[:cut], order[cut:]

    def fit_scaler(arr):
        flat = arr[train_idx].reshape(-1, arr.shape[-1])
        return flat.mean(0), flat.std(0) + 1e-6

    history_mean, history_std = fit_scaler(history_win)
    future_mean, future_std = fit_scaler(future_win)
    residual_mean, residual_std = fit_scaler(residual_win)
    standardize = lambda arr, mean, std: ((arr - mean) / std).astype(np.float32)

    return WindowedData(
        history_features=torch.tensor(standardize(history_win, history_mean, history_std)),
        future_features=torch.tensor(standardize(future_win, future_mean, future_std)),
        residual_target=torch.tensor(standardize(residual_win, residual_mean, residual_std)),
        history_feature_names=history_cols,
        future_feature_names=future_cols + ["lead_frac"],
        train_idx=train_idx, val_idx=val_idx,
        arome_forecast_wind=arome_wind, arome_forecast_gust=arome_gust,
        observed_wind=obs_wind, observed_gust=obs_gust,
        persistence_wind=pers_wind, persistence_gust=pers_gust,
        window_start_times=start_time,
        history_mean=history_mean, history_std=history_std,
        future_mean=future_mean, future_std=future_std,
        residual_mean=residual_mean, residual_std=residual_std,
        history_hours=history_hours, horizon_hours=horizon_hours,
    )
