from tokenizer import encode, decode
from model import TinyModel


# Đọc dữ liệu
with open("data/data.txt", "r", encoding="utf-8") as file:
    text = file.read()


# Tạo model
vocab_size = 9
model = TinyModel(vocab_size)


# Học nhiều lần
for epoch in range(10):

    for line in text.splitlines():

        ids = encode(line)

        for i in range(len(ids) - 1):

            current = ids[i]
            answer = ids[i + 1]

            prediction = model.learn(current, answer)


# In ra các weights sau khi học
print("=== WEIGHTS SAU KHI HỌC ===")

for i, row in enumerate(model.weights):
    word = decode([i])
    print(word, ":", row)


# Thử dự đoán
print("\n=== DỰ ĐOÁN ===")

word = "tôi"
current = encode(word)[0]

prediction = model.predict(current)

print(word, "→", decode([prediction]))

