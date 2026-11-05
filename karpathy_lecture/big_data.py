""" chứng minh rằng không quan tâm bao nhiêu outputs , chỉ cần sum lại và sau đó dựa vào Loss tổng và cập nhật params , output sẽ tự được điều chỉnh theo hướng giảm loss """

from nn import MLP

# mạng lưới neuron vẫn giữ nguyên như bài giảng 
mlp = MLP(3, [4, 4, 1])

# bộ data set lớn 
large_data_set = [
    (-1.0, 2.0, 3.2),
    (0.7, -1.4, 2.8),
    (3.1, 0.2, -0.9),
    (-2.6, 1.8, 0.5),
    (1.3, -3.2, 2.1),
    (0.4, 2.7, -1.6),
    (-0.8, 3.5, 1.2),
    (2.4, -0.3, -2.1),
    (-1.7, 0.9, 3.6),
    (3.3, -2.5, 0.6),
]

# mục tiêu ngẫu nhiên cho mỗi input trong bộ data set lớn
targets = [
    0.42,
    -1.73,
    2.16,
    0.85,
    -0.39,
    1.64,
    -2.08,
    0.27,
    1.31,
    -0.96,
]

learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    parameters = mlp.parameters()
    for parameter in parameters:
        parameter.grad = 0

    # forward pass
    # each predition is computed for each input in the large data set
    # put it in a list and then sum the loss for all predictions
    predictions = [mlp(inputs) for inputs in large_data_set]
    loss = sum(
        (prediction - target) ** 2
        for prediction, target in zip(predictions, targets)
    )
    loss.backward()

    for parameter in parameters:
        parameter.data -= learning_rate * parameter.grad

    outputs = [mlp(inputs) for inputs in large_data_set]
    total_loss = sum(
        (output - target) ** 2
        for output, target in zip(outputs, targets)
    )

    print(f"Vong {epoch + 1}/{epochs}")
    print("Parameters:", [round(parameter.data, 4) for parameter in parameters])
    print("Outputs:", [round(output.data, 4) for output in outputs])
    print(f"Total loss: {total_loss.data:.6f}\n")

