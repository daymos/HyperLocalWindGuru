"""Hyperparameters and run configuration."""

from dataclasses import dataclass


@dataclass
class Config:
    # windowing
    history_hours: int = 10      # observed hours fed to the encoder
    horizon_hours: int = 24      # forecast hours produced by the decoder
    # model
    hidden: int = 32             # GRU hidden size (lean: less overfitting)
    # optimization
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 60
    patience: int = 12           # early-stopping patience on val loss
    # split / misc
    train_frac: float = 0.8      # chronological train fraction
    seed: int = 0
    device: str = "cpu"
