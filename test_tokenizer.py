from tokenizer import encode, decode


text = "tôi ăn cơm"

ids = encode(text)

print("Câu ban đầu:")
print(text)

print("\nSau khi mã hóa:")
print(ids)

print("\nSau khi giải mã:")
print(decode(ids))

