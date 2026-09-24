"""Real 32x32 neural image generation using a learned decoder."""

import io
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ImageDecoder(nn.Module):
    def __init__(self, latent=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, 128 * 4 * 4),
            nn.GELU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)

def make_training_set(count=256):
    rng = np.random.default_rng(7)
    images = []
    for _ in range(count):
        image = Image.new("RGB", (32, 32), (int(rng.integers(0, 40)),) * 3)
        draw = ImageDraw.Draw(image)
        color = tuple(int(x) for x in rng.integers(80, 255, size=3))
        x, y = int(rng.integers(2, 18)), int(rng.integers(2, 18))
        size = int(rng.integers(8, 15))
        if rng.random() < 0.5:
            draw.rectangle((x, y, x + size, y + size), fill=color)
        else:
            draw.ellipse((x, y, x + size, y + size), fill=color)
        images.append(np.asarray(image, dtype=np.float32) / 255.0)
    return torch.tensor(np.stack(images), dtype=torch.float32).permute(0, 3, 1, 2)

class ImageGenerator:
    def __init__(self):
        self.model = ImageDecoder().to(DEVICE)
        self._train()

    def _train(self):
        data = make_training_set().to(DEVICE)
        rng = torch.Generator(device=DEVICE).manual_seed(11)
        z = torch.randn(data.size(0), 32, generator=rng, device=DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-3)

        self.model.train()
        for _ in range(80):
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
        image = self.model(z)[0].permute(1, 2, 0).cpu().numpy()
        pixels = np.clip(image * 255, 0, 255).astype(np.uint8)

        png = io.BytesIO()
        Image.fromarray(pixels, "RGB").save(png, format="PNG")
        grid = pixels.mean(axis=2).round().astype(int).tolist()

        return {
            "png": png.getvalue(),
            "width": 32,
            "height": 32,
            "grid": grid,
            "device": DEVICE,
        }
