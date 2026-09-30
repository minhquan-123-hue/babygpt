"""BabyGPT Step 10: Context -> Token.

Step 09 used one token to predict the next token.

That creates a problem:

    "tôi" -> what comes next?

The answer can depend on more than one word.

For example:

    "tôi ăn"   -> "cơm"
    "tôi uống" -> "nước"

So Step 10 gives the model a small context:
two previous tokens -> next token.

We still use a simple score table.
No neural network or attention yet.

The important new idea is:

    context -> scores -> next token
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


class ContextPredictor:
    def __init__(self, vocab_size):
        # Store scores for each two-token context.
        self.scores = {}

    def learn(self, context, next_token):
        # Create a score row when we see a new context.
        if context not in self.scores:
            self.scores[context] = [0 for _ in range(vocab_size)]

        # Increase the score of the correct next token.
        self.scores[context][next_token] += 1

    def predict(self, context):
        # Get the scores for this context.
        row = self.scores[context]

        # Choose the next token with the highest score.
        return max(range(len(row)), key=lambda i: row[i])


text = """
tôi ăn cơm
tôi uống nước
tôi ăn cơm
"""

vocab_size = len(module.word_to_id)
model = ContextPredictor(vocab_size)

# Learn from two tokens of context.
for line in text.strip().splitlines():
    tokens = encode(line)

    for i in range(len(tokens) - 2):
        context = (tokens[i], tokens[i + 1])
        next_token = tokens[i + 2]

        model.learn(context, next_token)


context = tuple(encode("tôi ăn"))
prediction = model.predict(context)

print("context   :", decode(list(context)))
print("prediction:", decode([prediction]))
