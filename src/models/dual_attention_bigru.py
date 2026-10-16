"""
src/models/dual_attention_bigru.py

Proposed model: Spectral-Gated Dual-Attention BiGRU (SGDA-BiGRU).

Pipeline (CNN-free, pure recurrent + attention, per the project plan's
architecture diagram):

    x (batch, T, C)
      -> GatedChannelAttention   ("where to look": per-sensor-channel weights,
                                   optionally modulated by a spectral prior)
      -> x_weighted = x * channel_weights
      -> BiGRU (stacked, bidirectional)
      -> TemporalAttention        ("when to look": per-timestep weights)
      -> context vector
      -> FC head -> logits

Novelty (see docs/research_gap_notes.md for the full writeup):
    Plain "channel attention + temporal attention + BiGRU" is a CLOSED gap
    (Dual-INABA, Scientific Reports, Dec 2025, implements exactly this on
    UCI-HAR/WISDM/PAMAP2). This model adds a genuinely new component on top
    of that closed architecture: SpectralChannelGate (in
    src/attention_modules.py) injects a frequency-domain periodicity prior
    (low-band energy ratio + spectral entropy, via a single cheap rFFT) into
    the channel-attention gate. No paper in the literature review computes
    channel attention from anything but learned time-domain statistics. This
    is not a re-run of an existing idea; it is a new, well-defined mechanism
    with an ablation switch (`use_spectral_gate`) to isolate its contribution
    against the already-known "plain dual attention" variant.

Ablation flags (all default True/True/True -- the full proposed model):
    use_channel_attention : if False, channel weights are all 1 (identity) --
                             isolates the temporal-attention-only contribution.
    use_temporal_attention: if False, falls back to mean-pooling over time
                             (like bigru_baseline) -- isolates the channel-
                             attention contribution.
    use_spectral_gate     : if False, channel attention is learned-only
                             (the "plain dual attention" / Dual-INABA-style
                             ablation arm). Only meaningful when
                             use_channel_attention=True.

Requires `sampling_rate_hz` at construction (differs per dataset: UCI-HAR
50Hz, WISDM 20Hz, PAMAP2 100Hz -- see src/preprocessing.py NATIVE_CONFIGS)
because the spectral gate needs real physical frequency units to define its
gait-band filter.
"""

import torch
import torch.nn as nn

from src.attention_modules import GatedChannelAttention, TemporalAttention
from src.models.stacked_rnn import StackedRNN


class DualAttentionBiGRU(nn.Module):
    """
    Input:  (batch, T, C)
    Output: (batch, num_classes) logits
    """

    def __init__(self, input_size: int, sampling_rate_hz: float,
                 hidden_size: int = 64, num_layers: int = 2, num_classes: int = 6,
                 dropout: float = 0.3, attention_dim: int = 64,
                 channel_reduction: int = 4, low_band_hz: tuple = (0.3, 3.5),
                 use_channel_attention: bool = True,
                 use_temporal_attention: bool = True,
                 use_spectral_gate: bool = True):
        super().__init__()
        self.use_channel_attention = use_channel_attention
        self.use_temporal_attention = use_temporal_attention

        if use_channel_attention:
            self.channel_attention = GatedChannelAttention(
                num_channels=input_size,
                sampling_rate_hz=sampling_rate_hz,
                reduction=channel_reduction,
                low_band_hz=low_band_hz,
                use_spectral_gate=use_spectral_gate,
            )

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

        if use_temporal_attention:
            self.temporal_attention = TemporalAttention(
                hidden_dim=hidden_size * 2, attention_dim=attention_dim,
            )

        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        channel_weights = None
        if self.use_channel_attention:
            channel_weights = self.channel_attention(x)          # (batch, C)
            x = x * channel_weights.unsqueeze(1)                  # (batch, T, C)

        out, _ = self.gru(x)                                      # (batch, T, hidden*2)

        temporal_weights = None
        if self.use_temporal_attention:
            context, temporal_weights = self.temporal_attention(out)  # (batch, hidden*2), (batch, T)
        else:
            context = out.mean(dim=1)                              # (batch, hidden*2)

        logits = self.head(context)
        if return_attention:
            return logits, {"channel_weights": channel_weights, "temporal_weights": temporal_weights}
        return logits


if __name__ == "__main__":
    # Smoke test across the three datasets' native (T, C, sampling_rate) shapes,
    # and across ablation flag combinations.
    configs = [
        (128, 6, 50, "ucihar"),
        (51, 3, 20, "wisdm"),
        (256, 12, 100, "pamap2"),
    ]
    ablations = [
        dict(use_channel_attention=True, use_temporal_attention=True, use_spectral_gate=True),
        dict(use_channel_attention=True, use_temporal_attention=True, use_spectral_gate=False),
        dict(use_channel_attention=True, use_temporal_attention=False, use_spectral_gate=True),
        dict(use_channel_attention=False, use_temporal_attention=True, use_spectral_gate=True),
    ]

    for T, C, sr, name in configs:
        dummy = torch.randn(8, T, C)
        for flags in ablations:
            model = DualAttentionBiGRU(input_size=C, sampling_rate_hz=sr, num_classes=6, **flags)
            out = model(dummy)
            logits, attn = model(dummy, return_attention=True)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{name:8s} flags={flags} -> output {tuple(out.shape)}, params={n_params}")
