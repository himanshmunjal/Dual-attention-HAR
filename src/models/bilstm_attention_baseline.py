"""
src/models/bilstm_attention_baseline.py

Baseline 2: BiLSTM -> single (temporal) additive attention -> FC head.
Reproduces the approach in literature review rows 1/2 (attention-based
BiLSTM with a single attention mechanism over time) -- NOT the dual
channel+temporal attention of the proposed model (row 5 / dual_attention_bigru).
Deliberately kept to one attention axis so the ablation story in the final
report is clean: bigru_baseline (no attention) -> this (one attention axis)
-> dual_attention_bigru (two attention axes, isolated and ablatable).
"""

import torch
import torch.nn as nn

from src.models.stacked_rnn import StackedRNN


class TemporalAttention(nn.Module):
    """
    Additive (Bahdanau-style) attention over the time dimension.
    Input:  (batch, T, hidden_dim)
    Output: context (batch, hidden_dim), weights (batch, T)
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        scores = self.score(x).squeeze(-1)        # (batch, T)
        weights = torch.softmax(scores, dim=1)     # (batch, T)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, hidden_dim)
        return context, weights


class BiLSTMAttentionBaseline(nn.Module):
    """
    Input:  (batch, T, C)
    Output: (batch, num_classes) logits

    Architecture: stacked BiLSTM -> temporal attention -> FC head.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 num_classes: int = 6, dropout: float = 0.3, attention_dim: int = 64):
        super().__init__()
        # stacked single-layer LSTMs with nn.Dropout between them == nn.LSTM(num_layers, dropout);
        # avoids cuDNN's built-in RNN dropout, which crashes the process at exit on Windows (see stacked_rnn.py)
        self.lstm = StackedRNN(
            "lstm",
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_dim=hidden_size * 2, attention_dim=attention_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        out, _ = self.lstm(x)                          # (batch, T, hidden*2)
        context, weights = self.attention(out)          # (batch, hidden*2), (batch, T)
        logits = self.head(context)
        if return_attention:
            return logits, weights
        return logits


if __name__ == "__main__":
    model = BiLSTMAttentionBaseline(input_size=6, num_classes=6)
    dummy = torch.randn(8, 128, 6)

    out = model(dummy)
    print("output shape:", out.shape)

    logits, weights = model(dummy, return_attention=True)
    print("attention weights shape:", weights.shape, "sum per sample:", weights.sum(dim=1)[:3])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable params:", n_params)