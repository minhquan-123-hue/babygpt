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

## Current step: 05 Prediction

A network produces several output values, or **scores**.

For example:

```
outputs = [0.2, 1.5, 0.7]
```

If the model must choose one answer, we need a simple rule:

```
prediction = output with the highest score
```

So here:

```
0.2   1.5   0.7
      ↑
    highest

prediction = 1
```

The important idea is:

**The network calculates scores. Prediction chooses the answer represented by the highest score.**

At this step there is still no training and no learning. We are only separating two ideas:

```
Network
  ↓
calculate scores
  ↓
Prediction
  ↓
choose highest score
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
