"""One training iteration + one generation on real clips from Greatest Hits.

Setup:
    uv venv && source .venv/bin/activate && uv pip install -r requirements.txt
Run:
    python run.py
"""

import io

import av
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import CLIPImageProcessor, CLIPVisionModel

from audio_encoder import TinyAudioEncoder
from codec import TinyCodec
from model import VideoToAudio

REPO = "dukesun99/GreatestHits-V2A"  # 2 s clips of a drumstick hitting/scratching things
CLIP = "openai/clip-vit-base-patch32"
N_CLIPS, N_FRAMES, SECONDS = 8, 8, 2.0
device = "cuda" if torch.cuda.is_available() else "cpu"


class ClipVLM(nn.Module):
    """Frozen CLIP vision tower as the given VLM: one pooled feature per frame."""

    def __init__(self):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(CLIP)
        self.dim = self.model.config.hidden_size

    def forward(self, video):  # [B, F, 3, 224, 224] -> [B, F, dim]
        B, frames = video.shape[:2]
        return self.model(pixel_values=video.flatten(0, 1)).pooler_output.view(B, frames, -1)


def read_video(data, processor):
    # decode the mp4, take N_FRAMES evenly spaced frames, CLIP preprocessing -> [F, 3, 224, 224]
    with av.open(io.BytesIO(data)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    idx = torch.linspace(0, len(frames) - 1, N_FRAMES).long()
    return processor(images=[frames[i] for i in idx], return_tensors="pt").pixel_values


def read_audio(data, sr):
    # decode, mono, resample to the codec rate, crop/pad to SECONDS -> [1, samples]
    wav, file_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav).mean(1)[None, None]
    wav = F.interpolate(wav, size=int(wav.size(-1) * sr / file_sr), mode="linear")[0]
    n = int(sr * SECONDS)
    return F.pad(wav, (0, max(0, n - wav.size(-1))))[..., :n]


# videos are in "queries", audio in "corpus", paired by id: q_<x> <-> c_<x>
videos = pq.read_table(hf_hub_download(REPO, "queries/test-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
audios = pq.read_table(hf_hub_download(REPO, "corpus/test-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
audios = {r["id"][2:]: r["audio"]["bytes"] for r in audios}

codec = TinyCodec()
processor = CLIPImageProcessor.from_pretrained(CLIP)
rows = videos[:N_CLIPS]
video = torch.stack([read_video(r["video"]["bytes"], processor) for r in rows]).to(device)
wav = torch.stack([read_audio(audios[r["id"][2:]], codec.sample_rate) for r in rows]).to(device)
print("video", tuple(video.shape), "wav", tuple(wav.shape))

# 8 frame tokens + 100 audio frames * 2 codebooks = 208 positions, fits max_tokens=256
vlm = ClipVLM()
model = VideoToAudio(vlm, codec, TinyAudioEncoder(), vlm_dim=vlm.dim).to(device)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)

model.train()
losses = model.losses(video, wav)
opt.zero_grad()
losses["total"].backward()
opt.step()
print("losses", {k: round(v.item(), 4) for k, v in losses.items()})

frames = int(SECONDS * codec.sample_rate / codec.hop)
out = model.generate(video[:1], audio_frames=frames, temperature=1.0)
sf.write("sample.wav", out[0, 0].cpu().numpy(), codec.sample_rate)
print("wrote sample.wav", tuple(out.shape))
