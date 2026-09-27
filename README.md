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
Token → Number
  ↓
Word Prediction
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

## Current steps: 08 Token → Number and 09 Word Prediction

### 08 Token → Number

A neural network works with numbers, not words.

So we give each word a token ID:

```
"tôi"  → 0
"ăn"   → 1
"cơm"  → 2
```

Then:

```
"tôi ăn cơm"
      ↓
[0, 1, 2]
```

Step 08 is only the bridge between human-readable words and numbers.

### 09 Word Prediction

Now we use those token numbers to predict the next word.

The model receives one token and gives a score to every possible next token:

```
"tôi"
  ↓
token 0
  ↓
scores for possible next tokens
  ↓
highest score
  ↓
"ăn"
```

The first version stores these scores in a simple table.

For example:

```
input "tôi"

"ăn"   → 2
"uống" → 1
"cơm"  → 0
```

The highest score becomes the prediction.

The important connection is now:

```
word
  ↓
token number
  ↓
scores
  ↓
prediction
  ↓
word
```

This is still deliberately simple. Later, we will replace the simple score table with a neural network that learns these scores.

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
