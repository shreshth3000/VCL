"""Video-to-audio model with generative and contrastive training branches."""

import torch
from torch import nn
from torch.nn import functional as F


class VideoToAudio(nn.Module):
    def __init__(
        self, vlm, codec, audio_encoder, vlm_dim=64, audio_dim=64,
        model_dim=64, contrastive_dim=64, contrastive_weight=0.1,
        max_tokens=256,
    ):
        super().__init__()
        self.vlm = vlm.eval().requires_grad_(False)
        self.codec = codec.eval().requires_grad_(False)
        self.audio_encoder = audio_encoder
        self.contrastive_weight = contrastive_weight
        self.books = codec.n_codebooks
        self.book_size = codec.codebook_size
        self.bos = self.books * self.book_size
        self.max_tokens = max_tokens

        # The two video projectors serve different objectives.
        self.audio_projector = nn.Linear(vlm_dim, model_dim)
        self.video_contrastive_projector = nn.Linear(vlm_dim, contrastive_dim)
        self.audio_contrastive_projector = nn.Linear(audio_dim, contrastive_dim)
        self.contrastive_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

        self.token_embedding = nn.Embedding(self.bos + 1, model_dim)
        self.position_embedding = nn.Embedding(max_tokens, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=4, dim_feedforward=4 * model_dim,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False,
        )
        self.output = nn.Linear(model_dim, self.book_size)

    def train(self, mode=True):
        super().train(mode)
        self.vlm.eval()
        self.codec.eval()
        return self

    def _predict(self, video_features, previous_tokens):
        """Video is a visible prefix; audio tokens can see only earlier tokens."""
        prefix = self.audio_projector(video_features)
        audio = self.token_embedding(previous_tokens)
        sequence = torch.cat((prefix, audio), dim=1)
        if sequence.size(1) > self.max_tokens:
            raise ValueError("Video plus audio tokens exceed max_tokens")

        positions = torch.arange(sequence.size(1), device=sequence.device)
        sequence = sequence + self.position_embedding(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            sequence.size(1), device=sequence.device,
        )
        hidden = self.transformer(sequence, mask=causal_mask)
        return self.output(hidden[:, prefix.size(1):])

    def losses(self, video, waveform):
        """Return both training objectives and their weighted sum."""
        if video.size(0) < 2:
            raise ValueError("InfoNCE needs at least two paired clips per batch")
        with torch.no_grad():
            features = self.vlm(video)
            codes = self.codec.encode(waveform)

        # Symmetric InfoNCE compares matching video/audio clips within a batch.
        audio_features = self.audio_encoder(waveform)
        video_global = features.mean(dim=1)
        audio_global = audio_features.mean(dim=1)
        z_video = F.normalize(self.video_contrastive_projector(video_global), dim=-1)
        z_audio = F.normalize(self.audio_contrastive_projector(audio_global), dim=-1)
        scale = self.contrastive_scale.exp().clamp(max=100)
        similarities = scale * (z_video @ z_audio.T)
        labels = torch.arange(video.size(0), device=video.device)
        contrastive = (
            F.cross_entropy(similarities, labels)
            + F.cross_entropy(similarities.T, labels)
        ) / 2

        # Teacher forcing teaches the Audio LM to generate the codec sequence.
        # Interleave codebooks by audio frame: (frame 0, book 0), (frame 0, book 1), ...
        target = codes.transpose(1, 2).flatten(1)
        book_ids = torch.arange(target.size(1), device=target.device) % self.books
        full_ids = target + book_ids * self.book_size
        bos = torch.full_like(full_ids[:, :1], self.bos)
        previous = torch.cat((bos, full_ids[:, :-1]), dim=1)
        logits = self._predict(features, previous)
        generation = F.cross_entropy(logits.flatten(0, 1), target.flatten())
        total = generation + self.contrastive_weight * contrastive
        return {"generation": generation, "contrastive": contrastive, "total": total}

    def loss(self, video, waveform):
        return self.losses(video, waveform)["total"]

    @torch.inference_mode()
    def generate(self, video, audio_frames, temperature=0.0):
        """Generate a fixed duration; 50 audio frames correspond to one second."""
        if audio_frames < 1 or temperature < 0:
            raise ValueError("audio_frames must be positive and temperature nonnegative")
        self.eval()
        features = self.vlm(video)
        previous = torch.full((video.size(0), 1), self.bos, device=video.device)
        generated = []

        for step in range(audio_frames * self.books):
            logits = self._predict(features, previous)[:, -1]
            if temperature == 0:
                index = logits.argmax(dim=-1)
            else:
                probabilities = F.softmax(logits / temperature, dim=-1)
                index = torch.multinomial(probabilities, 1).squeeze(1)
            generated.append(index)
            full_id = index + (step % self.books) * self.book_size
            previous = torch.cat((previous, full_id[:, None]), dim=1)

        codes = torch.stack(generated, dim=1)
        codes = codes.reshape(video.size(0), audio_frames, self.books)
        return self.codec.decode(codes.transpose(1, 2))
