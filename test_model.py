from tokenizer import encode, decode
from model import TinyModel


# Đọc dữ liệu
with open("data/data.txt", "r", encoding="utf-8") as file:
    text = file.read()


# Tạo model
vocab_size = 9
model = TinyModel(vocab_size)


# Cho model xem dữ liệu
print("=== DỮ LIỆU ===")
print(text)


# Hiện weights ban đầu
print("\n=== WEIGHTS BAN ĐẦU ===")
print(model.weights)


# Cho model dự đoán
word = "tôi"
word_id = encode(word)[0]

prediction_id = model.predict(word_id)
prediction = decode([prediction_id])

print("\n=== DỰ ĐOÁN ===")
print("Đầu vào:", word)
print("Dự đoán:", prediction)

