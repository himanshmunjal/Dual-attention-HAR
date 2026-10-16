"""
src/models/deepconvlstm.py

Baseline 4: DeepConvLSTM (Ordóñez & Roggen, "Deep Convolutional and LSTM
Recurrent Neural Networks for Multimodal Wearable Activity Recognition",
Sensors 16(1):115, 2016) -- the canonical CNN+RNN baseline in the HAR
literature, reimplemented here under the same training protocol as every
other model.

Kept as in the paper:
  - 4 conv layers, 64 filters each, kernel (5, 1): each filter slides along
    time only and is shared across sensor channels (no mixing between
    channels until the LSTM), ReLU, no padding, no pooling
  - 2 unidirectional LSTM layers with 128 units, dropout 0.5 on the LSTM
    inputs, classification from the last time step
Deliberately not tuned to this project's shared knobs (hidden_size=64,
dropout=0.3): it has its own constructor argument names, so build_model never
passes those in and the published configuration is what gets compared.
Per the paper it is larger than the other models; params/FLOPs are reported
alongside accuracy.
"""

import torch
import torch.nn as nn

from src.models.stacked_rnn import StackedRNN


class DeepConvLSTM(nn.Module):
    """
    Input:  (batch, T, C)
    Output: (batch, num_classes) logits

    The four unpadded convs shorten T by 4 * (kernel - 1) = 16 samples
    (WISDM: 51 -> 35, UCI-HAR/PAMAP2: 128 -> 112).
    """

    def __init__(self, input_size: int, num_classes: int = 6, num_filters: int = 64,
                 filter_size: int = 5, num_conv_layers: int = 4, lstm_units: int = 128,
                 num_lstm_layers: int = 2, dropout_p: float = 0.5):
        super().__init__()
        convs = []
        for i in range(num_conv_layers):
            convs += [nn.Conv2d(1 if i == 0 else num_filters, num_filters, kernel_size=(filter_size, 1)),
                      nn.ReLU()]
        self.convs = nn.Sequential(*convs)
        self.dropout = nn.Dropout(dropout_p)
        # stacked single-layer LSTMs with nn.Dropout between them == nn.LSTM(num_layers, dropout);
        # avoids cuDNN's built-in RNN dropout, which crashes the process at exit on Windows (see stacked_rnn.py)
        self.lstm = StackedRNN(
            "lstm",
            input_size=num_filters * input_size,
            hidden_size=lstm_units,
            num_layers=num_lstm_layers,
            bidirectional=False,
            dropout=dropout_p if num_lstm_layers > 1 else 0.0,
        )
        self.head = nn.Linear(lstm_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.convs(x.unsqueeze(1))          # (batch, filters, T', C)
        x = x.permute(0, 2, 1, 3).reshape(b, x.shape[2], -1)  # (batch, T', filters * C)
        out, _ = self.lstm(self.dropout(x))     # (batch, T', lstm_units)
        return self.head(self.dropout(out[:, -1]))


if __name__ == "__main__":
    for T, C, k, name in [(128, 6, 6, "ucihar"), (51, 3, 6, "wisdm"), (128, 18, 12, "pamap2")]:
        model = DeepConvLSTM(input_size=C, num_classes=k)
        out = model(torch.randn(8, T, C))
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:8s} T={T:>3} C={C:>2} -> output {tuple(out.shape)}, params={n_params:,}")
