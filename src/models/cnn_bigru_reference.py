"""
src/models/cnn_bigru_reference.py

Baseline 3: 2-3 Conv1D+pooling layers -> BiGRU -> FC head.
This is the "heavy" reference baseline the efficiency claim (headline result
#4 in the plan) is measured against: if Dual-Attention BiGRU can match or
approach this model's accuracy at meaningfully lower params/FLOPs/latency,
that's the paper's core finding either way.

Conv layers extract local temporal patterns (short-window motion motifs)
before the BiGRU models longer-range dependencies over the downsampled
sequence -- the standard CNN-then-RNN hybrid pattern from literature rows
7-10, reimplemented here under identical training conditions to every other
model in this project (not copied from a literature table).
"""

import torch
import torch.nn as nn

from src.models.stacked_rnn import StackedRNN


class CNNBiGRUReference(nn.Module):
    """
    Input:  (batch, T, C)
    Output: (batch, num_classes) logits

    Architecture: 3x Conv1D blocks (first two with max-pooling, halving the
    temporal length twice) -> BiGRU -> mean-pool over time -> FC head.

    Conv1d expects channel-first input, so we permute (batch, T, C) ->
    (batch, C, T) going in, and back to (batch, T', C') before the BiGRU.
    """

    def __init__(self, input_size: int, conv_channels=(64, 128, 128),
                 kernel_size: int = 5, hidden_size: int = 64, num_layers: int = 2,
                 num_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        c1, c2, c3 = conv_channels
        pad = kernel_size // 2

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_size, c1, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        # third conv layer, no pooling -- adds depth without shrinking short
        # sequences (e.g. WISDM's 51-sample windows) down too far
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)

        # stacked single-layer GRUs with nn.Dropout between them == nn.GRU(num_layers, dropout);
        # avoids cuDNN's built-in RNN dropout, which crashes the process at exit on Windows (see stacked_rnn.py)
        self.gru = StackedRNN(
            "gru",
            input_size=c3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)          # (batch, C, T)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)          # (batch, T', C')

        out, _ = self.gru(x)            # (batch, T', hidden*2)
        pooled = out.mean(dim=1)        # (batch, hidden*2)
        return self.head(pooled)


if __name__ == "__main__":
    # Smoke test across the three datasets' native window sizes -- the two
    # max-pools must not collapse the shortest sequence (WISDM, 51 samples).
    for T, C, name in [(128, 6, "ucihar"), (51, 3, "wisdm"), (256, 6, "pamap2")]:
        model = CNNBiGRUReference(input_size=C, num_classes=6)
        dummy = torch.randn(8, T, C)
        out = model(dummy)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:8s} T={T:>3} C={C} -> output {tuple(out.shape)}, params={n_params}")