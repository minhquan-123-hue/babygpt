"""BabyGPT Step 12: Attention.

Step 10 showed that context matters.
Step 11 gave each token a vector.

Now we combine those ideas.

A token should not always treat every other token
in the context as equally important.

Example:

    "tôi ăn cơm"

When we are trying to understand "cơm",
"ăn" may be more relevant than "tôi".

Attention gives the current token a way to
look at the other tokens and assign importance.

For this first step, we use simple similarity:

    similarity = query · key

Then:

    attention score
        ↓
    higher score = more relevant
        ↓
    weighted combination of value vectors

We use the same vector as query, key, and value
to keep the idea visible.

This is NOT the full Transformer attention yet.
There is no softmax, no learned Q/K/V matrices,
and no multi-head attention.

The important new idea is:

    token -> look at context -> weighted information
"""


from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


# Load token conversion from Step 08.
path = Path(__file__).with_name("08_token_to_number.py")
spec = spec_from_file_location("token_to_number", path)
module = module_from_spec(spec)
spec.loader.exec_module(module)

encode = module.encode


class Embedding:
    def __init__(self):
        # Small vectors from Step 11.
        self.vectors = {
            0: [0.2, 0.8],  # tôi
            1: [0.7, 0.1],  # ăn
            2: [0.6, 0.3],  # cơm
            3: [0.6, 0.2],  # uống
            4: [0.1, 0.9],  # nước
        }

    def get(self, token):
        return self.vectors[token]


def dot(a, b):
    # Dot product gives a simple measure of how similar
    # two vectors are in this tiny example.
    return sum(x * y for x, y in zip(a, b))


def attention(context, current_index, embedding):
    current = embedding.get(context[current_index])

    scores = []

    # Compare the current token with every token in the context.
    for token in context:
        vector = embedding.get(token)
        scores.append(dot(current, vector))

    # Convert scores into simple weights.
    total = sum(scores)
    weights = [score / total for score in scores]

    # Combine the context vectors using those weights.
    result = [0 for _ in current]

    for weight, token in zip(weights, context):
        vector = embedding.get(token)

        for i in range(len(result)):
            result[i] += weight * vector[i]

    return scores, weights, result


embedding = Embedding()

text = "tôi ăn cơm"
tokens = encode(text)

current_index = 2
scores, weights, result = attention(tokens, current_index, embedding)

print("context :", text)
print("looking :", "cơm")
print("scores  :", [round(score, 3) for score in scores])
print("weights :", [round(weight, 3) for weight in weights])
print("result  :", [round(value, 3) for value in result])
