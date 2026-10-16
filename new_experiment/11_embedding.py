"""BabyGPT Step 11: Embedding.

Step 08 turned words into token IDs:

    "tôi" -> 0
    "ăn"  -> 1

But token IDs are just labels.

The number 1 does not mean that "ăn" is more similar
to "tôi" than to "nước".

A model needs a richer representation.

An embedding gives each token a small vector:

    "tôi"   -> [0.2, 0.8]
    "ăn"    -> [0.7, 0.1]
    "uống"  -> [0.6, 0.2]

Now a token is represented by several numbers instead
of one arbitrary ID.

For this step, the vectors are just a lookup table.
We are not training them yet.

The important new idea is:

    token ID -> vector
"""


from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


# Load the vocabulary from Step 08.
path = Path(__file__).with_name("08_token_to_number.py")
spec = spec_from_file_location("token_to_number", path)
module = module_from_spec(spec)
spec.loader.exec_module(module)

encode = module.encode


class Embedding:
    def __init__(self):
        # Each token ID points to a small vector.
        #
        # These numbers are intentionally simple.
        # Real embeddings are learned from data.
        self.vectors = {
            0: [0.2, 0.8],  # tôi
            1: [0.7, 0.1],  # ăn
            2: [0.6, 0.3],  # cơm
            3: [0.6, 0.2],  # uống
            4: [0.1, 0.9],  # nước
        }

    def get(self, token):
        # Look up the vector belonging to a token.
        return self.vectors[token]


embedding = Embedding()

text = "tôi ăn cơm"
tokens = encode(text)

print("text  :", text)
print("tokens:", tokens)

for token in tokens:
    vector = embedding.get(token)
    print(token, "->", vector)
