from pathlib import Path

import torch


# Tìm file tên theo vị trí thư mục hiện tại để chạy được dù ở thư mục khác.
names_path = Path(__file__).parent.parent / "makemore" / "names.txt"
# Đọc file và chia thành từng dòng tên.
words = names_path.read_text(encoding="utf-8").splitlines()

# Lấy tất cả ký tự có trong các tên, sắp xếp để tạo thứ tự nhất quán.
chars = sorted(set("".join(words)))
# Gán mỗi ký tự một số, bắt đầu từ 1.
stoi = {character: index + 1 for index, character in enumerate(chars)}
# Dùng dấu chấm để đánh dấu bắt đầu và kết thúc tên.
stoi["."] = 0
# Tạo bảng ngược để đổi số trở lại thành ký tự.
itos = {index: character for character, index in stoi.items()}

# Số ký tự trước chữ hiện tại dùng để đoán chữ tiếp theo.
block_size = 3
X = []
Y = []

# Mỗi tên tạo ra các cặp dữ liệu đầu vào và ký tự cần dự đoán.
for word in words[:5]:
    # Hiển thị tên đang được xử lý.
    # print(word)
    # Bắt đầu bằng ba số 0, mỗi số biểu thị dấu chấm.
    context = [0] * block_size
    for character in word + ".":
        # Đổi ký tự cần dự đoán thành số.
        index = stoi[character]
        # Lưu ba ký tự trước đó làm đầu vào và ký tự hiện tại làm kết quả mong muốn.
        X.append(context)
        Y.append(index)
        # Chuyển ba số ngữ cảnh thành ký tự để dễ đọc.
        context_text = "".join(itos[item] for item in context)
        # print(context_text, "--->", itos[index])
        # Bỏ ký tự cũ nhất và thêm ký tự mới để chuẩn bị cho bước tiếp theo.
        context = context[1:] + [index]

# Chuyển dữ liệu sang tensor để dùng trong mô hình PyTorch.
X = torch.tensor(X)
Y = torch.tensor(Y)

# print(f"{X.shape[0]} examples")

# tạo ra một cỗ máy chọn số ngẫu nhiên
generator = torch.Generator().manual_seed(2147483647)
# kích thước chiều 
embedding_size = 2
# 1 hidden layer duy nhất có 100 neurons
hidden_size = 100

# kích thước của C[27x2]
C = torch.randn((len(stoi), embedding_size), generator=generator)
# kích thước của W1[6x100]
W1 = torch.randn((block_size * embedding_size, hidden_size), generator=generator)
# kích thước của b1[1x100]
b1 = torch.randn(hidden_size, generator=generator)
# kích thước của W2[100x27]
W2 = torch.randn((hidden_size, len(stoi)), generator=generator)
# kích thước của b2[1x27]
b2 = torch.randn(len(stoi), generator=generator)

# tổng tham số của model
parameters = [C, W1, b1, W2, b2]
print(sum(parameter.numel() for parameter in parameters))

# torch kết hợp tensor với tensor thành dạng
# tensor[tensors] ví dụ cụ thể là 32x3x2
emb = C[X]
# tanh là hàm nén biến dữ liệu đầu ra chỉ trải dài từ -1 đến 1
# torch biến 32x3x2 về dạng 32x6 để thực hiện @ với weight 1 6x100
h = torch.tanh(emb.reshape(X.shape[0], block_size * embedding_size) @ W1 + b1)
# sau khi có kết quả đầu ra từ hidden layer
# tiếp tục @ với output layer số neuron == vocab size
# để tạo ra logits tức dữ liệu của ngài neural net
logits = h @ W2 + b2
# chuyển dữ liệu của ngài neural net thành dữ liệu mà
# con người chúng ta có thể hiểu được
""" counts = logits.exp()
# chuyển đổi phần dữ liệu hiểu được về xác suất
prob = counts / counts.sum(1, keepdim=True)
# dùng hàm biểu diễn sai số nlls để tạo ra một
# con số cuối cùng guild toàn bộ ngài neural net
# để cập nhật tham số giảm loss
# trong quá trình giảm loss , probalities gán cho những
# label đúng sẽ được tự động cập nhật tăng lên
loss = -prob[torch.arange(X.shape[0]), Y].log().mean() """

loss = F.cross_entrophy(logits, Y)
print(loss)
