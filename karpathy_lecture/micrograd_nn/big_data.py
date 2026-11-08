inputs = [1.0, 2.0]
targets = [-2.0, 3.0]

weight = -2.0
# weight_grad = 1.0 * output1_grad + 1.0 * output2_grad
bias = 1.6
# bias_grad = 1.0 * final_output1_grad + 1.0 * final_output2_grad


output1 = inputs[0] * weight
# output1_grad = 1.0 * final_output1_grad

output2 = inputs[1] * weight
# output2_grad = 1.0 * final_output2_grad

final_output1 = output1 + bias
# final_output1_grad = 2(final_output1 - targets[0]) * loss1_grad
final_output2 = output2 + bias
# final_output2_grad = 2(final_output2 - targets[1]) * loss2_grad

loss1 = (final_output1 - targets[0]) ** 2
# loss1_grad = 1.0 * tt_loss_grad 
loss2 = (final_output2 - targets[1]) ** 2
# loss2_grad = 1.0 * tt_loss_grad 

total_loss = loss1 + loss2
# tt_loss_grad = 1.0 

learning_rate = 0.1

# sau nhieu lan hoc
# cac phan weight se duoc cap nhat theo cong thuc: para.data += -learning_rate * param.grad
# cung voi input duoc dua vao thi weight va bias van se giu nguyen
# va output van se tinh toan gan dat den muc theo mong muon
