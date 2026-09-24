"""Real 3-second waveform generation using a learned neural decoder."""

import io
import wave
import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 8000
SECONDS = 3
SAMPLES = SAMPLE_RATE * SECONDS

class AudioDecoder(nn.Module):
    def __init__(self, latent=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, 256),
            nn.GELU(),
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Linear(1024, SAMPLES),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)

def make_training_set(count=32):
    t = torch.linspace(0, SECONDS, SAMPLES)
    rows = []
    for i in range(count):
        f1, f2 = 110 + i * 13, 220 + i * 7
        signal = 0.55 * torch.sin(2 * torch.pi * f1 * t)
        signal += 0.25 * torch.sin(2 * torch.pi * f2 * t)
        rows.append(signal)
    return torch.stack(rows)

class AudioGenerator:
    def __init__(self):
        self.model = AudioDecoder().to(DEVICE)
        self._train()

    def _train(self):
        data = make_training_set().to(DEVICE)
        rng = torch.Generator(device=DEVICE).manual_seed(17)
        z = torch.randn(data.size(0), 32, generator=rng, device=DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-3)

        self.model.train()
        for _ in range(40):
            prediction = self.model(z)
            loss = ((prediction - data) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.model.eval()

    @torch.no_grad()
    def generate(self, seed=0):
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        z = torch.randn(1, 32, generator=generator, device=DEVICE)
        waveform = self.model(z)[0].cpu().numpy()
        pcm = np.clip(waveform * 32767, -32768, 32767).astype(np.int16)

        wav = io.BytesIO()
        with wave.open(wav, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            output.writeframes(pcm.tobytes())

        debug = pcm[::max(1, len(pcm) // 256)].astype(int).tolist()
        return {
            "wav": wav.getvalue(),
            "duration_seconds": SECONDS,
            "sample_rate": SAMPLE_RATE,
            "amplitudes": debug,
            "device": DEVICE,
        }
