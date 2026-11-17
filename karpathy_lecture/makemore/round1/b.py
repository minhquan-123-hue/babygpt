"""model voi nhieu input la nhung cai ten , chu khong chi la hieu mo toan hoc co ban nua, bat dau co the xu ly voi data that"""

import torch
import torch.nn.functional as F

# lấy toàn bộ dữ liệu thành từng từ
words = open('names.txt', 'r').read().splitlines() 
# ép thành 26 ký tự 
chars = sorted(list(set(''.join(words))))


# tạo map cho từ sang số
# số sang từ
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}




Matrix = torch.zeros(27,27,dtype=torch.int32)
for word in words:
    characters = ['.'] + list(word) + ['.']
    for character1, character2 in zip(characters, characters[1:]):
        ix1 = stoi[character1]
        ix2 = stoi[character2]
        Matrix[ix1][ix2] += 1
P = (Matrix+1).float()
P /= P.sum(1, keepdim=True)





g = torch.Generator().manual_seed(2147483647)
for i in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=False, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    ''.join(out)





log_likelihood = 0.0
n = 0
for word in words[:3]:
    characters = ['.'] + list(word) + ['.']
    for character1, character2 in zip(characters, characters[1:]):
        ix1 = stoi[character1]
        ix2 = stoi[character2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
nll = -log_likelihood
loss =  nll/n






