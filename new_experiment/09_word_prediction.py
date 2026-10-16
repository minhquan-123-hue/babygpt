"""BabyGPT Step 09: Word Prediction.

Now we connect tokens to prediction.

The model receives one token ID and produces a score
for every possible next token.

Example:

    input token: "tôi"

    scores:
    "ăn"   -> 3
    "uống" -> 1
    "cơm"  -> 0

The highest score is the prediction:

    "ăn"

For this first word-prediction step, the scores are
stored in a simple table.

This is intentionally small. Later, a neural network
will learn to produce these scores instead of storing
one table row for every input token.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


# Load the token conversion from Step 08.
path = Path(__file__).with_name("08_token_to_number.py")
spec = spec_from_file_location("token_to_number", path)
module = module_from_spec(spec)
spec.loader.exec_module(module)

encode = module.encode
decode = module.decode


class WordPredictor:
    def __init__(self, vocab_size):
        # One row for each input token.
        # Each row contains a score for every possible next token.
        self.scores = [
            [0 for _ in range(vocab_size)]
            for _ in range(vocab_size)
        ]

    def predict(self, token):
        # Choose the next token with the highest score.
        row = self.scores[token]
        return max(range(len(row)), key=lambda i: row[i])

    def learn(self, token, next_token):
        # Increase the score of the next token we want.
        self.scores[token][next_token] += 1


text = """
tôi ăn cơm
tôi uống nước
tôi ăn cơm
"""

vocab_size = len(module.word_to_id)
model = WordPredictor(vocab_size)

# Learn which token tends to come after each token.
for line in text.strip().splitlines():
    tokens = encode(line)

    for i in range(len(tokens) - 1):
        current = tokens[i]
        next_token = tokens[i + 1]

        model.learn(current, next_token)


word = "tôi"
token = encode(word)[0]
prediction = model.predict(token)

print("input     :", word)
print("token     :", token)
print("prediction:", decode([prediction]))
