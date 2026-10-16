"""BabyGPT Step 12: Attention v4, learnable scoring.

Each token compares itself with every token using a tiny set
of trainable dimension weights. Changing those weights changes
the scores, attention weights, and context representations.

This keeps v3's simple score normalization. No softmax,
Q/K/V projections, Transformer blocks, or multiple heads.
"""


def weighted_dot(a, b, scoring_parameters):
    """Score two vectors, with one learnable weight per dimension."""
    return sum(
        parameter * x * y
        for parameter, x, y in zip(scoring_parameters, a, b)
    )


def attention_for_token(embeddings, current_index, scoring_parameters):
    current = embeddings[current_index]

    scores = [
        weighted_dot(current, other, scoring_parameters)
        for other in embeddings
    ]

    # Keep the simple normalization used in v3.
    total = sum(scores)
    weights = [score / total for score in scores]

    representation = [0.0 for _ in current]
    for weight, vector in zip(weights, embeddings):
        for i, value in enumerate(vector):
            representation[i] += weight * value

    return {
        "scores": scores,
        "weights": weights,
        "representation": representation,
    }


def attention(embeddings, scoring_parameters):
    """Return one result (scores, weights, vector) for every token."""
    return [
        attention_for_token(embeddings, i, scoring_parameters)
        for i in range(len(embeddings))
    ]


if __name__ == "__main__":
    embeddings = [
        [0.2, 0.8],  # tôi
        [0.7, 0.1],  # ăn
        [0.6, 0.3],  # cơm
    ]

    # These are the tiny learnable parameters, one per vector dimension.
    scoring_parameters = [1.0, 1.0]

    for label, parameters in [
        ("Initial parameters", [1.0, 1.0]),
        ("Changed parameters", [10.0, 0.1]),
    ]:
        print(label, parameters)
        results = attention(embeddings, parameters)
        for i, result in enumerate(results):
            print(
                f"token {i}:",
                "weights =", [round(x, 3) for x in result["weights"]],
                "context =", [round(x, 3) for x in result["representation"]],
            )
        print()
