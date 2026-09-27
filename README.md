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

## Current step: 04 Network

Step 03 gave us one layer:

```
many inputs
     ↓
   layer
     ↓
many outputs
```

But one layer is still limited. We want to process the result again.

A network connects multiple layers:

```
inputs
  ↓
layer 1
  ↓
layer 2
  ↓
outputs
```

The important idea is simple:

**The output of one layer becomes the input of the next layer.**

For example:

```
[2, 3]
   ↓
Layer 1
   ↓
[output1, output2]
   ↓
Layer 2
   ↓
[output1, output2]
```

So a network is not a completely new kind of calculation. It is mainly a way to **connect layers together**.

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
