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
    dead = (df["wind_avg_kt"] == 0) & (df["wind_max_kt"] == 0)
    return df[~dead].reset_index(drop=True), int(dead.sum())


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic (sin/cos) encodings for hour-of-day and every wind direction."""
    df = df.copy()
    hr = df["datetime"].dt.hour.values
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    for col, s, c in [
        ("wind_dir_deg", "obs_dir_sin", "obs_dir_cos"),
        ("arome_wind_dir_deg", "fc_dir_sin", "fc_dir_cos"),
        ("ukv_wind_dir_deg", "ukv_dir_sin", "ukv_dir_cos"),
        ("icond2_wind_dir_deg", "icond2_dir_sin", "icond2_dir_cos"),
    ]:
        rad = np.deg2rad(df[col].astype(float).values)
        df[s], df[c] = np.sin(rad), np.cos(rad)
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
    Xh: torch.Tensor          # (N, H, n_seq_all) standardized history features
    Xf: torch.Tensor          # (N, F, n_fut_all) standardized future features (incl. lead)
    Y: torch.Tensor           # (N, F, 2) standardized residual target [wind, gust]
    seq_names: List[str]
    fut_names: List[str]
    train_idx: np.ndarray
    val_idx: np.ndarray
    Aw: np.ndarray            # AROME forecast wind  (N, F)
    Ag: np.ndarray            # AROME forecast gust  (N, F)
    OW: np.ndarray            # observed wind truth  (N, F)
    OG: np.ndarray            # observed gust truth  (N, F)
    Pw: np.ndarray            # persistence wind     (N, F)
    Pg: np.ndarray            # persistence gust     (N, F)
    times: np.ndarray         # target window start time (N,)
    seq_mean: np.ndarray
    seq_std: np.ndarray
    fut_mean: np.ndarray
    fut_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    H: int
    F: int

    def channel_indices(self, which: str):
        """Column indices into Xh / Xf for a named channel set ('arome' / 'multimodel')."""
        ch = CHANNELS[which]
        seq_idx = torch.tensor([self.seq_names.index(c) for c in ch["seq"]])
        fut_idx = torch.tensor([self.fut_names.index(c) for c in ch["fut"]])
        return seq_idx, fut_idx


def build_windows(df: pd.DataFrame, cfg) -> WindowedData:
    """Slide (H history + F horizon) windows within contiguous hourly runs, split
    chronologically, and standardize on training statistics."""
    H, F = cfg.history_hours, cfg.horizon_hours
    seq_all, fut_feat = SEQ_BASE + EXTRA, FUT_BASE + EXTRA

    is_gap = np.r_[True, df["datetime"].diff().dt.total_seconds().values[1:] != 3600]
    run_id = np.cumsum(is_gap)

    seqM, futM = df[seq_all].values.astype(np.float32), df[fut_feat].values.astype(np.float32)
    ow, og = df["wind_avg_kt"].values.astype(np.float32), df["wind_max_kt"].values.astype(np.float32)
    fw, fg = df["arome_wind_10m_kt"].values.astype(np.float32), df["arome_gust_10m_kt"].values.astype(np.float32)
    times, lead = df["datetime"].values, ((np.arange(F) + 1) / F).astype(np.float32)

    Xh, Xf, Y, Aw, Ag, OW, OG, Pw, Pg, T = ([] for _ in range(10))
    for _, idx in pd.Series(np.arange(len(df))).groupby(run_id):
        idx = idx.values
        for i in range(len(idx) - (H + F) + 1):
            hs, fu = idx[i:i + H], idx[i + H:i + H + F]
            Xh.append(seqM[hs]); Xf.append(np.c_[futM[fu], lead])
            Y.append(np.stack([ow[fu] - fw[fu], og[fu] - fg[fu]], 1))
            Aw.append(fw[fu]); Ag.append(fg[fu]); OW.append(ow[fu]); OG.append(og[fu])
            Pw.append(np.full(F, ow[hs[-1]], np.float32)); Pg.append(np.full(F, og[hs[-1]], np.float32))
            T.append(times[fu[0]])
    Xh, Xf, Y = np.array(Xh), np.array(Xf), np.array(Y, np.float32)
    Aw, Ag, OW, OG = np.array(Aw), np.array(Ag), np.array(OW), np.array(OG)
    Pw, Pg, T = np.array(Pw), np.array(Pg), np.array(T)

    order = np.argsort(T); cut = int(len(order) * cfg.train_frac)
    tr, va = order[:cut], order[cut:]

    sm, ss = Xh[tr].reshape(-1, Xh.shape[2]).mean(0), Xh[tr].reshape(-1, Xh.shape[2]).std(0) + 1e-6
    fm, fs = Xf[tr].reshape(-1, Xf.shape[2]).mean(0), Xf[tr].reshape(-1, Xf.shape[2]).std(0) + 1e-6
    ym, ysd = Y[tr].reshape(-1, 2).mean(0), Y[tr].reshape(-1, 2).std(0) + 1e-6
    z = lambda a, m, s: ((a - m) / s).astype(np.float32)

    return WindowedData(
        Xh=torch.tensor(z(Xh, sm, ss)), Xf=torch.tensor(z(Xf, fm, fs)), Y=torch.tensor(z(Y, ym, ysd)),
        seq_names=seq_all, fut_names=fut_feat + ["lead_frac"],
        train_idx=tr, val_idx=va,
        Aw=Aw, Ag=Ag, OW=OW, OG=OG, Pw=Pw, Pg=Pg, times=T,
        seq_mean=sm, seq_std=ss, fut_mean=fm, fut_std=fs, y_mean=ym, y_std=ysd,
        H=H, F=F,
    )
