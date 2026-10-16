"""
src/models/bigru_baseline.py

Baseline 1: Plain BiGRU -> FC head. No attention, no CNN.
Serves as the floor against which every other model (BiLSTM+attention,
CNN+BiGRU reference, Dual-Attention BiGRU) is compared on both accuracy
and efficiency (params/FLOPs/latency).
"""

import torch
import torch.nn as nn

from src.models.stacked_rnn import StackedRNN


class BiGRUBaseline(nn.Module):
    """
    Input:  (batch, T, C)  -- T timesteps, C sensor channels
    Output: (batch, num_classes) logits

    Architecture: stacked BiGRU -> mean-pool over time -> FC head.
    Mean-pooling (rather than taking only the last timestep) is used because
    the last output position mixes the *forward* direction's final state
    with the *backward* direction's state from processing only the first
    timestep -- mean-pooling is a more standard, less bug-prone way to get a
    fixed-size representation from a bidirectional RNN for classification.
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, num_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        # stacked single-layer GRUs with nn.Dropout between them == nn.GRU(num_layers, dropout);
        # avoids cuDNN's built-in RNN dropout, which crashes the process at exit on Windows (see stacked_rnn.py)
        self.gru = StackedRNN(
            "gru",
            input_size=input_size,
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
        out, _ = self.gru(x)          # (batch, T, hidden*2)
        pooled = out.mean(dim=1)      # (batch, hidden*2)
        return self.head(pooled)      # (batch, num_classes)


if __name__ == "__main__":
    # Smoke test with dummy input matching e.g. UCI-HAR native config (128, 6)
    model = BiGRUBaseline(input_size=6, num_classes=6)
    dummy = torch.randn(8, 128, 6)
    out = model(dummy)
    print("output shape:", out.shape)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable params:", n_params)