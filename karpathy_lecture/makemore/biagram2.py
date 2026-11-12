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





xs, ys = [], []
for word in words[:1]:
    characters = ['.'] + list(word) + ['.']
    for character1, character2 in zip(characters, characters[1:]):
        ix1 = stoi[character1]
        ix2 = stoi[character2]
        print(character1, character2)
        xs.append(ix1)
        ys.append(ix2)




xs = torch.tensor(xs)
ys = torch.tensor(ys)





g = torch.Generator().manual_seed(2147483647)
W = torch.randn(27,27)
xenc = F.one_hot(xs, num_classes=27).float()
logits = xenc @ W
counts = logits.exp()
probs = counts / counts.sum(1,keepdim=True)





print(probs)