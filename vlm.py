"""Small CLIP-style visual tower standing in for the supplied pretrained VLM."""

import torch
from torch import nn
from torch.nn import functional as F


class TinyCLIPVisionEncoder(nn.Module):
    """Encode each video frame with patches, a class token, and self-attention.

    Input: [batch, frames, 3, height, width], with pixels in [0, 1].
    Output: [batch, frames, feature_dim]. The audio model treats these frame
    embeddings as its video prefix. This tower has random, frozen weights.
    """

    def __init__(self, image_size=32, patch_size=8, width=64, feature_dim=64):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        patch_count = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(3, width, patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.position_embedding = nn.Parameter(torch.randn(1, patch_count + 1, width) * 0.02)
        self.pre_norm = nn.LayerNorm(width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=4 * width,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False,
        )
        self.post_norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, feature_dim)

    def forward(self, video):
        batch, frames, channels, height, width = video.shape
        if channels != 3:
            raise ValueError("Video must have three color channels")

        images = video.reshape(batch * frames, channels, height, width)
        images = F.interpolate(images, size=self.image_size, mode="bilinear", align_corners=False)
        patches = self.patch_embedding(images).flatten(2).transpose(1, 2)
        class_tokens = self.class_token.expand(batch * frames, -1, -1)
        tokens = torch.cat((class_tokens, patches), dim=1)
        tokens = self.pre_norm(tokens + self.position_embedding)
        encoded = self.transformer(tokens)
        frame_features = self.projection(self.post_norm(encoded[:, 0]))
        return frame_features.reshape(batch, frames, -1)
