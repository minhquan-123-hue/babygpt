"""model voi nhieu input la nhung cai ten , chu khong chi la hieu mo toan hoc co ban nua, bat dau co the xu ly voi data that"""

import torch

words = open('names.txt', 'r').read().splitlines() 
    
# N la so lan tan suat 2 tu noi lien
# gap nhau   
N = torch.zeros(27,27,dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}


for word in words:
    characters = ['.'] + list(word) + ['.']
    for character1, character2 in zip(characters, characters[1:]):
        ix1 = stoi[character1]
        ix2 = stoi[character2]
        N[ix1][ix2] += 1

print(N[0, :])

