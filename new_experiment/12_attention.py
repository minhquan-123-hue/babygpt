"""BabyGPT Step 12: Attention v2.

Step 12 v1 let one token look at the whole context.

But attention is more useful when every token does this.

For example:

    "tôi ăn cơm"

Instead of creating one representation for only "cơm":

    "tôi" -> context representation
    "ăn"  -> context representation
    "cơm" -> context representation

Each token looks at all tokens, decides how relevant
they are, and creates its own context representation.

We still keep the mechanism simple:

    token vector
        ↓
    compare with every token vector
        ↓
    scores
        ↓
    weights
        ↓
    weighted sum
        ↓
    context representation

This is still simplified attention.
We use the same vector as query, key, and value.
There is no softmax, learned Q/K/V, or multi-head attention yet.
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
    # Compare two vectors with a dot product.
    return sum(x * y for x, y in zip(a, b))


def attention_for_token(context, current_index, embedding):
    current = embedding.get(context[current_index])

    scores = []

    # This token looks at every token in the context.
    for token in context:
        vector = embedding.get(token)
        scores.append(dot(current, vector))

    # Turn scores into simple weights.
    total = sum(scores)
    weights = [score / total for score in scores]

    # Gather information from the whole context.
    result = [0 for _ in current]

    for weight, token in zip(weights, context):
        vector = embedding.get(token)

        for i in range(len(result)):
            result[i] += weight * vector[i]

    return scores, weights, result


def attention(context, embedding):
    # Every token creates its own context representation.
    representations = []

    for current_index in range(len(context)):
        scores, weights, result = attention_for_token(
            context,
            current_index,
            embedding,
        )

        representations.append({
            "scores": scores,
            "weights": weights,
            "representation": result,
        })

    return representations


embedding = Embedding()

text = "tôi ăn cơm"
tokens = encode(text)

results = attention(tokens, embedding)

print("context:", text)
print()

for i, result in enumerate(results):
    print("token", i)
    print("scores :", [round(score, 3) for score in result["scores"]])
    print("weights:", [round(weight, 3) for weight in result["weights"]])
    print("result :", [round(value, 3) for value in result["representation"]])
    print()
