"""Tiny neural codec with the same encode/decode interface as a real codec.

Its weights are random. It demonstrates codec tokens, not useful sound quality.
"""

import torch
from torch import nn


class TinyCodec(nn.Module):
    """Represent each 320-sample audio frame with two residual code indices."""

    sample_rate = 16000
    hop = 320

    def __init__(self, codebook_size=32, dim=32, n_codebooks=2):
        super().__init__()
        self.codebook_size = codebook_size
        self.n_codebooks = n_codebooks
        self.codebooks = nn.Parameter(torch.randn(n_codebooks, codebook_size, dim) * 0.5)

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 4, stride=4), nn.GELU(),
            nn.Conv1d(16, dim, 4, stride=4), nn.GELU(),
            nn.Conv1d(dim, dim, 4, stride=4), nn.GELU(),
            nn.Conv1d(dim, dim, 5, stride=5),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(dim, dim, 5, stride=5), nn.GELU(),
            nn.ConvTranspose1d(dim, dim, 4, stride=4), nn.GELU(),
            nn.ConvTranspose1d(dim, 16, 4, stride=4), nn.GELU(),
            nn.ConvTranspose1d(16, 1, 4, stride=4), nn.Tanh(),
        )

    @torch.no_grad()
    def encode(self, waveform):
        """Map [B, 1, samples] to integer codes [B, K, audio_frames]."""
        if waveform.size(-1) < self.hop:
            raise ValueError("Waveform must contain at least 320 samples")
        usable = waveform.size(-1) // self.hop * self.hop
        residual = self.encoder(waveform[..., :usable]).transpose(1, 2)
        codes = []
        for book in self.codebooks:
            index = torch.cdist(residual, book).argmin(dim=-1)
            codes.append(index)
            residual = residual - book[index]
        return torch.stack(codes, dim=1)

    @torch.no_grad()
    def decode(self, codes):
        """Map integer codes [B, K, audio_frames] to a waveform."""
        if codes.size(1) != self.n_codebooks:
            raise ValueError("Wrong number of codec codebooks")
        quantized = torch.zeros(
            codes.size(0), codes.size(2), self.codebooks.size(-1),
            device=codes.device,
        )
        for book_index, book in enumerate(self.codebooks):
            quantized = quantized + book[codes[:, book_index]]
        return self.decoder(quantized.transpose(1, 2))
