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

print(words[:8])
print(f"{len(words)} words")
print(itos)

# Số ký tự trước chữ hiện tại dùng để đoán chữ tiếp theo.
block_size = 3
X = []
Y = []

# Mỗi tên tạo ra các cặp dữ liệu đầu vào và ký tự cần dự đoán.
for word in words[:5]:
    # Hiển thị tên đang được xử lý.
    print(word)
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
        print(context_text, "--->", itos[index])
        # Bỏ ký tự cũ nhất và thêm ký tự mới để chuẩn bị cho bước tiếp theo.
        context = context[1:] + [index]

# Chuyển dữ liệu sang tensor để dùng trong mô hình PyTorch.
X = torch.tensor(X)
Y = torch.tensor(Y)
print(X)
print(Y)

print(f"{X.shape[0]} examples")