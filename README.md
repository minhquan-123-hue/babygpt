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

## Current step: 02 Many neurons

One neuron gives us one output.

The next problem is:

```
What if we need several outputs from the same input?
```

Step 02 solves this by using several independent neurons.

Each neuron has:
- its own weight
- its own bias
- the same input

So the data flow is:

```
                 ┌─ neuron 1 → output 1
input ───────────┼─ neuron 2 → output 2
                 └─ neuron 3 → output 3
```

The implementation still uses plain Python only. No PyTorch, NumPy, TensorFlow, or web UI is needed.

## Old experiment

The first tiny word-prediction experiment is kept as reference in:

```
old_experiment/
    tokenizer.py
    model.py
    train.py
    test_model.py
    test_tokenizer.py
    data/
        data.txt
```

This code is **not part of the current learning path**.

Do not refactor it or add features to it unless we explicitly decide to revisit it. It is kept because it shows an earlier, simpler attempt at word prediction.

## Principle

If a component is too complicated to explain with a few concrete numbers, simplify it before moving on.
