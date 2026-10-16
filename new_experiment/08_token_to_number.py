"""BabyGPT Step 08: Token -> Number.

A neural network works with numbers, not words.

So before a word can enter the network, we give each word
a number called a token ID.

Example:

    "tôi"  -> 0
    "ăn"   -> 1
    "cơm"  -> 2

This step only converts between words and numbers.
"""


word_to_id = {
    "tôi": 0,
    "ăn": 1,
    "cơm": 2,
    "uống": 3,
    "nước": 4,
}


id_to_word = {
    value: key
    for key, value in word_to_id.items()
}


def encode(text):
    # Words -> token IDs
    return [word_to_id[word] for word in text.split()]


def decode(ids):
    # Token IDs -> words
    return " ".join(id_to_word[i] for i in ids)


text = "tôi ăn cơm"

tokens = encode(text)

print("text  :", text)
print("tokens:", tokens)
print("back  :", decode(tokens))
