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
Context → Token
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

## Current steps: 08 Token → Number, 09 Word Prediction, 10 Context → Token, and 11 Embedding

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

### 10 Context → Token

Step 09 used only one token as input.

But the next word can depend on the words before it:

```
"tôi ăn"   → "cơm"
"tôi uống" → "nước"
```

Step 10 gives the predictor two previous tokens as context:

```
context
   ↓
scores for possible next tokens
   ↓
highest score
   ↓
next token
```

The first version is still a simple score table. We are deliberately not adding attention yet.

The important new connection is:

```
one token → next token
        ↓
context → next token
```

This prepares the problem that attention will eventually solve: deciding which parts of a larger context matter for the next prediction.

### 11 Embedding

Step 08 gave every word an integer ID.

But these IDs are only labels:

```
"tôi" → 0
"ăn"  → 1
"cơm" → 2
```

The number `1` does not contain useful meaning about the word `"ăn"`.

We need a representation that contains several numbers:

```
"tôi" → [0.2, 0.8]
"ăn"  → [0.7, 0.1]
"cơm" → [0.6, 0.3]
```

This is an **embedding**.

The model can now work with a small vector instead of an arbitrary token ID:

```
token ID
   ↓
embedding lookup
   ↓
vector
```

For this first step, the vectors are just a small lookup table. They are not learned yet.

Later, learning will adjust these numbers from data so that the vectors become useful representations.

The important connection is:

```
word
  ↓
token ID
  ↓
vector
  ↓
neural network
```

Embedding solves a different problem from context.

- **Context** tells us which previous tokens are available.
- **Embedding** gives each token a useful numerical representation.

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
