"""model voi nhieu input la nhung cai ten , chu khong chi la hieu mo toan hoc co ban nua, bat dau co the xu ly voi data that"""

import torch

# lấy toàn bộ dữ liệu thành từng từ
words = open('names.txt', 'r').read().splitlines() 
# ép thành 26 ký tự 
chars = sorted(list(set(''.join(words))))


# tạo map cho từ sang số
# số sang từ
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# tạo tensor 27x27 dựa vào số lượng chữ cái
# column 0 là từ kết thúc , row 0 là từ bắt đầu
Matrix = torch.zeros(27,27,dtype=torch.int32)

# tạo vòng lặp đi theo từng từ trong toàn bộ từ
# nối dấu "." ở đầu và dấu "." ở cuối
for word in words:
    characters = ['.'] + list(word) + ['.']
    for character1, character2 in zip(characters, characters[1:]):
        ix1 = stoi[character1]
        ix2 = stoi[character2]
        Matrix[ix1][ix2] += 1


P = Matrix.float()
P /= P.sum(1, keepdim=True)

# cỗ máy tạo số ngẫu nhiên
g = torch.Generator().manual_seed(2147483647)

# dự đoán 5 từ liên tiếp
for i in range(5):
    # tạo danh sách để nối từng chữ đơn lẻ 
    out = []
    ix = 0
    while True:
        p = P[ix]
        # sau khi có dữ liệu xác suất từ 1 row của 27 column
        # bằng 1 cách nào đó torch.multinomial biết các chọn ra một trong
        # những xác suất cao của row đó một cách "ngẫu nhiên"
        # nhưng nó không có dữ liệu xa hơn như vậy nên là nó đó mà chỉ dựa vào 1 từ trước nó
        ix = torch.multinomial(p, num_samples=1, replacement=False, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    ''.join(out)


for word in words[:3]:
    characters = ['.'] + list(word) + ['.']
    for character1, character2 in zip(characters, characters[1:]):
        ix1 = stoi[character1]
        ix2 = stoi[character2]
        prob = P[ix1, ix2]
        print(f"{character1}{character2}: {prob*100:.2f}%")