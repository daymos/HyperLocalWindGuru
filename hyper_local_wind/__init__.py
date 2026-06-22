"""
hyper_local_wind — site-specific AROME wind-forecast correction.

A seq2seq GRU that, from the last N hours of observations + multi-model NWP
forecasts, predicts a residual correction to AROME's next-F-hour wind & gust
forecast. The notebook drives this package (load -> train -> predict -> evaluate).
"""

from .config import Config
from .crypto import (
    read_encrypted_parquet,
    write_encrypted_parquet,
    encrypt_bytes,
    decrypt_bytes,
)
from .data import (
    prepare,
    build_windows,
    WindowedData,
    CHANNELS,
)
from .model import Seq2SeqCorrector
from .train import train_model
from .inference import predict, save_model, load_model
from .evaluate import comparison_table, perlead_rmse

__all__ = [
    "Config",
    "read_encrypted_parquet", "write_encrypted_parquet", "encrypt_bytes", "decrypt_bytes",
    "prepare", "build_windows", "WindowedData", "CHANNELS",
    "Seq2SeqCorrector", "train_model", "predict", "save_model", "load_model",
    "comparison_table", "perlead_rmse",
]
