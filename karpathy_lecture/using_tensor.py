import torch

# ở từng mảng đa chiều , tạo 1 con số vô hướng, tạo đạo hàm local của nó là 0.0
x1 = torch.Tensor([2.0]).double();x1.requires_grad = True
x2 = torch.Tensor([0.0]).double();x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double();w1.requires_grad = True   
w2 = torch.Tensor([1.0]).double();w2.requires_grad = True
b = torch.Tensor([6.8337373363827]).double();b.requires_grad = True

# sau đó khi mà được thực hiện toàn bộ phép tính rồi 
n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

print(o.data.item())
# nó có được output cuối cùng
# sau đó sẽ tạo ra đạo hàm cho output 
# lan truyền ngược lại cho từng grad của từng input [local node derivatives (xác định _op) * next node derivative]
# cộng dồn nếu input được thực hiện trên nhiều phép tính 
o.backward()

print('----')
print('x1.grad:', x1.grad.item())
print('x2.grad:', x2.grad.item())
print('w1.grad:', w1.grad.item())
print('w2.grad:', w2.grad.item())
print('b.grad:', b.grad.item())