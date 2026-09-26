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

## Current step: 03 Layer

Step 02 had:

```
one input
   ↓
many neurons
   ↓
many outputs
```

But a real neuron also needs to accept **many inputs**.

For example:

```
inputs = [2, 3]
```

One neuron can combine them:

```
2 × weight1 + 3 × weight2 + bias
```

Step 03 puts these ideas together:

```
                 ┌─ neuron 1 → output 1
input 1 ─────────┼─ neuron 2 → output 2
input 2 ─────────┼─ neuron 3 → output 3
                 └────────────
```

Every neuron receives the same inputs, but each neuron has its own weights and bias.

So a **layer is simply many neurons processing the same input vector**.

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
