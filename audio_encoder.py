"""Small trainable audio encoder used only by the InfoNCE branch."""

from torch import nn


class TinyAudioEncoder(nn.Module):
    """Map mono waveform samples to a sequence of audio features."""

    def __init__(self, feature_dim=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, 9, stride=4, padding=4), nn.GELU(),
            nn.Conv1d(16, 32, 9, stride=4, padding=4), nn.GELU(),
            nn.Conv1d(32, feature_dim, 9, stride=4, padding=4), nn.GELU(),
        )

    def forward(self, waveform):
        # [B, 1, samples] -> [B, roughly samples / 64, feature_dim]
        return self.layers(waveform).transpose(1, 2)
