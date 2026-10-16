"""
src/models/stacked_rnn.py

StackedRNN: a multi-layer (Bi)GRU / (Bi)LSTM built from single-layer RNNs
with nn.Dropout between layers -- the same computation as
nn.GRU / nn.LSTM(num_layers=N, dropout=p), which applies dropout to the
outputs of every layer except the last, with identical parameter shapes.

Why not nn.GRU(num_layers=N, dropout=p) directly: on Windows with
torch 2.11.0+cu128 (cuDNN 9.19), using cuDNN's built-in RNN dropout in
training mode makes the Python process crash at exit (0xC0000409, shown as
exit 127 in Git Bash). The run itself completes and its results are written,
but the non-zero exit code makes run_parallel.py / run_all.sh treat every
job as failed. Single-layer cuDNN RNNs plus a regular nn.Dropout avoid
cuDNN's dropout state entirely.
"""

import torch
import torch.nn as nn

_CELLS = {"gru": nn.GRU, "lstm": nn.LSTM}


class StackedRNN(nn.Module):
    """
    Input:  (batch, T, input_size)
    Output: (out, None) -- out is (batch, T, hidden_size * num_directions).
            The second element mirrors nn.GRU/nn.LSTM's (output, h_n) tuple so
            call sites can keep `out, _ = self.gru(x)`; final hidden states
            are not used by any model here.
    """

    def __init__(self, cell: str, input_size: int, hidden_size: int, num_layers: int = 1,
                 bidirectional: bool = False, dropout: float = 0.0):
        super().__init__()
        rnn = _CELLS[cell]
        out_size = hidden_size * (2 if bidirectional else 1)
        self.layers = nn.ModuleList(
            rnn(input_size=input_size if i == 0 else out_size, hidden_size=hidden_size,
                batch_first=True, bidirectional=bidirectional)
            for i in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            if i > 0:
                x = self.dropout(x)
            x, _ = layer(x)
        return x, None
