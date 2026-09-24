# BabyGPT

A small educational multimodal generative AI laboratory.

## Current build

- Text: a tiny GPT-style causal Transformer trained on data/data.txt, generating up to 200 words.
- Image: a real neural decoder generating a 32x32 RGB image from a latent vector.
- Audio: a real neural decoder generating a 3-second 8 kHz waveform.
- UI: visualizes text as a neural network, audio as integer amplitudes, and image as a 32x32 integer grid.

The UI is theatrical, but the backend is not. Generated artifacts come from PyTorch models.

## Run

    pip install -r requirements.txt
    python app.py

Then open http://127.0.0.1:5000.

## Limitation

This is a learning system, not a competitive foundation model. The image and audio generators use tiny synthetic datasets so their architecture and generation process remain understandable and runnable on a laptop.

The next stage should improve training persistence, conditioning, datasets, evaluation, and model architecture rather than replacing the backend with external generation APIs.
