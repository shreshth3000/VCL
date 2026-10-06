"""End-to-end smoke test: video tensor -> codec tokens -> output.wav.

Run: .venv/bin/python test.py
Or:  .venv/bin/python test.py --input my_video.pt --output my_audio.wav
The optional input file must contain a [frames, 3, height, width] tensor in [0, 1].
"""

import argparse
import struct
import wave

import torch

from audio_encoder import TinyAudioEncoder
from codec import TinyCodec
from model import VideoToAudio
from vlm import TinyCLIPVisionEncoder


def save_wav(path, waveform, sample_rate):
    samples = waveform.squeeze().clamp(-1, 1)
    samples = (samples * 32767).to(torch.int16).cpu().tolist()
    pcm = struct.pack(f"<{len(samples)}h", *samples)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Path to a video tensor saved with torch.save")
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--check-loss", action="store_true", help="Check both losses without training")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(1)
    if args.input:
        video = torch.load(args.input, map_location="cpu", weights_only=True)
    else:
        video = torch.rand(4, 3, 32, 32)
    if video.ndim != 4 or video.shape[1] != 3:
        raise ValueError("Input must have shape [frames, 3, height, width]")

    codec = TinyCodec()
    model = VideoToAudio(TinyCLIPVisionEncoder(), codec, TinyAudioEncoder())
    waveform = model.generate(video.unsqueeze(0).float(), audio_frames=10)
    assert waveform.shape == (1, 1, 10 * codec.hop)
    assert torch.isfinite(waveform).all()
    save_wav(args.output, waveform, codec.sample_rate)
    print(f"Wrote {args.output}: {waveform.size(-1)} samples at {codec.sample_rate} Hz")
    print("The models have random weights; the audio is only a pipeline check.")

    if args.check_loss:
        # Two pairs provide in-batch negatives for InfoNCE. No optimizer step runs.
        paired_video = torch.stack((video, video.flip(0))).float()
        paired_audio = torch.rand(2, 1, 10 * codec.hop)
        model.train()
        losses = model.losses(paired_video, paired_audio)
        losses["total"].backward()
        assert all(torch.isfinite(value) for value in losses.values())
        assert model.video_contrastive_projector.weight.grad is not None
        assert model.audio_projector.weight.grad is not None
        print("Loss check:", {name: round(value.item(), 4) for name, value in losses.items()})


if __name__ == "__main__":
    main()
