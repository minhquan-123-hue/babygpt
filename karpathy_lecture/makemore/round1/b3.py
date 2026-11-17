from pathlib import Path
import torch
import torch.nn.functional as F



names_path = Path(__file__).resolve().parent / "names.txt"
with names_path.open("r", encoding="utf-8") as file:
    words = [line.strip() for line in file if line.strip()]

chars = sorted({char for word in words for char in word})

stoi = {char: index + 1 for index, char in enumerate(chars)}
stoi["."] = 0
itos = {index: char for char, index in stoi.items()}

xs, ys = [], []
for word in words[:1]:
    characters = ["."] + list(word) + ["."]
    for char1, char2 in zip(characters, characters[1:]):
        xs.append(stoi[char1])
        ys.append(stoi[char2])

xs = torch.tensor(xs)
print(xs)
ys = torch.tensor(ys)
print(ys)

generator = torch.Generator().manual_seed(2147483647)
weight_matrix = torch.randn((len(itos), len(itos)), generator=generator, requires_grad=True)

x_encoded = F.one_hot(xs, num_classes=len(itos)).float()
learning_rate = 0.1

for step in range(100):
    # Forward pass: tính loss giữa dự đoán và ký tự tiếp theo thực tế.
    logits = x_encoded @ weight_matrix
    loss = F.cross_entropy(logits, ys) + 0.01*(weight_matrix**2).mean()

    # Backward pass và cập nhật trọng số theo hướng làm giảm loss.
    weight_matrix.grad = None
    loss.backward()
    with torch.no_grad():
        weight_matrix -= learning_rate * weight_matrix.grad

    if step % 10 == 0 or step == 99:
        print(f"lần học {step + 1}: loss = {loss.item():.4f}")