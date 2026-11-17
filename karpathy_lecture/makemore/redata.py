import torch

"""chia nhỏ thành từng file để mô tả được luồng quá trình mà cụ karpathy thực hiện"""

# tổng số từ khoảng hơn 32 ngàn từ
# từ ngắn nhất 2
# từ dài nhất 15
words = open("names.txt", 'r').read().splitlines()

# tạo tensor chứa counts
N = torch.zeros((27,27), dtype=torch.int32)

# tạo bảng vocab
chars = sorted(list(set(''.join(words))))
# map từ sang số , để sử dụng index truy cập matrix counts
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# tạo 2 list và va so sánh 2 ký tự liền kề
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        row = stoi[ch1]
        col = stoi[ch2]

        N[row,col] += 1

print(N)


