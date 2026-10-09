"""BabyGPT Step 12: Attention v3.

Input: a sequence of token embedding vectors.
Output: one context-aware vector for every input token.

Each token compares its vector with every vector in the
sequence, turns the scores into simple weights, then
takes a weighted sum of the input vectors.

This reuses the simple attention idea from v2.
No Q/K/V projections, softmax, or multi-head attention.
"""


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def attention_for_token(embeddings, current_index):
    current = embeddings[current_index]

    # Compare this token with every token in the sequence.
    scores = [dot(current, other) for other in embeddings]

    # Simple score normalization, matching the v2 approach.
    total = sum(scores)
    weights = [score / total for score in scores]

    # Combine the input vectors using these weights.
    result = [0.0 for _ in current]

    for weight, vector in zip(weights, embeddings):
        for i, value in enumerate(vector):
            result[i] += weight * value

    return result


def attention(embeddings):
    # Every input token gets its own context-aware vector.
    return [
        attention_for_token(embeddings, i)
        for i in range(len(embeddings))
    ]


if __name__ == "__main__":
    embeddings = [
        [0.2, 0.8],  # "tôi"
        [0.7, 0.1],  # "ăn"
        [0.6, 0.3],  # "cơm"
    ]

    context_vectors = attention(embeddings)

    print("Input embeddings:")
    for vector in embeddings:
        print([round(value, 3) for value in vector])

    print("\nContext representations:")
    for vector in context_vectors:
        print([round(value, 3) for value in vector])
