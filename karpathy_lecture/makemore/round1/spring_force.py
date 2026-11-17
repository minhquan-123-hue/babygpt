""" dùng trường hợp vô lý nhưng có chứa ý tưởng tăng loss => làm giảm weight nhiều hơn để có độ mịn hơn ???"""

# hiện tại thì ở đây ta chỉ cần nhiều vào function này để thấy được
# weight_grad được cộng dồn độ ảnh hưởng grad

# forward pass
w1 = -2
w1_grad = 0.0

i = 2
i_grad = 0.0
b = 2
b_grad = 0.0

o1 = w1 * i
o1_grad = 0.0
nll = o1 + b
nll_grad = 0.0

force = 0.01
force_grad = 0.0

spring_force = w1 * force
spring_force_grad = 0.0

total_loss = nll + spring_force
print("L",total_loss)


# backward pass
total_loss_grad = 1.0

spring_force_grad = 1.0 * total_loss_grad
nll_grad = 1.0 * total_loss_grad

w1_grad += force * spring_force_grad
print("wg",w1_grad)
force_grad = w1 * spring_force_grad

o1_grad = 1.0 * nll_grad
b_grad = 1.0 * nll_grad

i_grad = w1 * o1_grad
w1_grad += i * o1_grad
print("wg",w1_grad)

# learning 
w1 += 0.1 * w1_grad

# forward pass 2, sau khi học xong thì sẽ 
# cập nhật tham số trong mô hình 
# dẫn đến việc loss sẽ giảm
w1_grad = 0.0

i = 2
i_grad = 0.0
b = 2
b_grad = 0.0

o1 = w1 * i
o1_grad = 0.0
nll = o1 + b
nll_grad = 0.0

force = 0.01
force_grad = 0.0

spring_force = w1 * force
spring_force_grad = 0.0

total_loss = nll + spring_force
print("L",total_loss)

