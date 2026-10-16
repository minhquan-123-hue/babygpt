"""BabyGPT Step 06: loss.

Step 05 gave us a prediction.

Now we need to answer:

    How wrong was the prediction?

For this simple step, we measure the distance between
the predicted score and the correct score.

    error = (prediction_score - target_score) ** 2

A small error means the prediction is close to the target.
A large error means the prediction is far from the target.

We are measuring the error only.
The model does not learn yet.
"""


def neuron(inputs, weights, bias):
    total = sum(input_value * weight for input_value, weight in zip(inputs, weights))
    return total + bias


def layer(inputs, neurons):
    return [
        neuron(inputs, weights, bias)
        for weights, bias in neurons
    ]


def network(inputs, layers):
    output = inputs

    for current_layer in layers:
        output = layer(output, current_layer)

    return output


def predict(outputs):
    # Choose the output with the highest score.
    return max(range(len(outputs)), key=lambda i: outputs[i])


def loss(prediction_score, target_score):
    # Squared error turns the distance into a positive number.
    return (prediction_score - target_score) ** 2


inputs = [2, 3]

layers = [
    [
        ([0.5, 1], 1),
        ([2, -1], 0),
    ],
    [
        ([1, 2], 0),
        ([-1, 1], 1),
        ([0.5, -0.5], 0),
    ],
]

outputs = network(inputs, layers)
prediction = predict(outputs)

# Suppose the correct answer is output 2.
target = 2

prediction_score = outputs[prediction]
target_score = outputs[target]

error = loss(prediction_score, target_score)

print("inputs         :", inputs)
print("outputs        :", outputs)
print("prediction     :", prediction)
print("target         :", target)
print("prediction score:", prediction_score)
print("target score   :", target_score)
print("error          :", error)
