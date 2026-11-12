"""small operation to illustrate how backpropagation works"""

n1 = -2
n1_grad = 2

n2 = 1
n2_grad = -4

n3 = n1 * n2
n3_grad = 2

n4 = 2
n4_grad = 2

n5 = n3 + n4
n5_grad = 2

output = n5
target = -1 #ta muon dieu chinh param la n1 , n4 dua theo grad hien tai cua no de tien dan toi target 

loss = (output - target) ** 2 # ket qua se la 2 va duoc truyen dao ve cho n5 
loss_grad = 2 * (output - target) 

for i in range(20):
    # backpropagate 
    n5_grad = loss_grad
    n4_grad = n5_grad * 1
    n3_grad = n5_grad * 1

    n2_grad = n3_grad * n1
    n1_grad = n3_grad * n2

    # update params
    learning_rate = 0.01
    n1 += -learning_rate * n1_grad
    n4 += -learning_rate * n4_grad

    # recompute forward pass
    n3 = n1 * n2
    n5 = n3 + n4
    output = n5
    loss = (output - target) ** 2

    print(f"Iteration {i}: loss={loss}, output={output}, target={target}, n1={n1}, n4={n4}")
