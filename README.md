# BabyGPT

BabyGPT is being rebuilt from the smallest neural-network ideas upward.

## Learning path

```
Neuron
  ↓
Many neurons
  ↓
Layer
  ↓
Network
  ↓
Prediction
  ↓
Loss
  ↓
Learning
  ↓
Token prediction
  ↓
Embedding
  ↓
Attention
  ↓
Transformer
  ↓
Tiny GPT
```

Each new component should appear because it solves a problem from the previous step.

## Current step: 01 Neuron

```
output = input × weight + bias
```

The implementation uses plain Python only. No PyTorch, NumPy, TensorFlow, or web UI is needed at this stage.

## Original experiments

The repository keeps the first tiny word-prediction experiment as reference:

- `tokenizer.py`
- `model.py`
- `train.py`
- `test_model.py`
- `test_tokenizer.py`
- `data/data.txt`

The previous multimodal prototype is no longer the learning path. We will rebuild it deliberately from fundamentals instead of jumping directly to a Transformer.

## Principle

If a component is too complicated to explain with a few concrete numbers, simplify it before moving on.
