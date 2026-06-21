"""Encoder-decoder GRU that predicts a per-hour [wind, gust] residual."""

import torch
import torch.nn as nn


class Seq2SeqCorrector(nn.Module):
    """Encode the H-hour history; run a decoder over the F future forecast vectors
    seeded with that context; a shared MLP head emits a [wind, gust] residual per hour."""

    def __init__(self, n_seq: int, n_fut: int, hidden: int = 32):
        super().__init__()
        self.encoder = nn.GRU(n_seq, hidden, batch_first=True)
        self.decoder = nn.GRU(n_fut, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 2)
        )

    def forward(self, hist: torch.Tensor, fut: torch.Tensor) -> torch.Tensor:
        _, context = self.encoder(hist)        # (1, B, hidden)
        out, _ = self.decoder(fut, context)    # (B, F, hidden)
        return self.head(out)                  # (B, F, 2)
