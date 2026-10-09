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

### 12 Attention

Step 11 gave each token a vector.

Now we need a way for one token to look at the other tokens in its context.

For example, when looking at `"cơm"` in:

```
"tôi ăn cơm"
```

not every token needs to be equally important.

Attention compares the current token with the tokens around it:

```
current token
     ↓
compare with context
     ↓
attention scores
     ↓
attention weights
     ↓
weighted context information
```

In this first implementation, similarity is measured with a dot product:

```
similarity = query · key
```

We then use the scores to make a weighted combination of the context vectors.

To keep the idea simple, this version uses the same embedding vector as query, key, and value. It does not use softmax, learned Q/K/V matrices, or multiple heads yet.

The important new connection is:

```
token
  ↓
embedding
  ↓
look at context
  ↓
weighted information
```

This is the core problem that attention solves.

### 12 Attention v2

Attention v1 let one selected token look at the context.

Attention v2 extends that idea to every token.

For:

```
"tôi ăn cơm"
```

we now create three context representations:

```
"tôi" → looks at the context → representation
"ăn"  → looks at the context → representation
"cơm" → looks at the context → representation
```

Each token follows the same process:

```
token vector
    ↓
compare with every context vector
    ↓
attention scores
    ↓
attention weights
    ↓
weighted sum
    ↓
that token's context representation
```

So attention is no longer:

```
context → one selected representation
```

It becomes:

```
context
  ↓
every token looks at the context
  ↓
one context representation per token
```

This is still simplified attention. The same embedding vector is used as query, key, and value. Softmax, learned Q/K/V projections, and multi-head attention come later.

### 12 Attention v3

Attention v3 takes a sequence of embedding vectors directly and returns one new context-aware vector for each token.

Example input:

```
[
    [0.2, 0.8],  # tôi
    [0.7, 0.1],  # ăn
    [0.6, 0.3],  # cơm
]
```

Each input vector attends to all vectors, then combines them using its own weights. The output has the same number of vectors as the input.

Run the example:

```bash
python 12_attention_v3.py
```

Run the test:

```bash
python -m unittest test_attention_v3.py
```

The test checks that every input token gets a separate output vector, that the vector dimensions are preserved, and that the example tokens receive different context representations.

This is still the project's simple attention mechanism: dot-product scores and simple normalization. It does not add softmax or learned Q/K/V projections.

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
