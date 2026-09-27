"""BabyGPT Step 05: prediction.

A network can produce several output values.
But if we want the model to choose one answer,
we need a simple prediction rule.

For this step:

    prediction = output with the highest score

Example:

    outputs = [0.2, 1.5, 0.7]

The model predicts:

    index 1

because 1.5 is the highest score.

We are not training anything yet.
The network simply produces scores, and prediction chooses one.
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

print("inputs   :", inputs)
print("outputs  :", outputs)
print("prediction:", prediction)
