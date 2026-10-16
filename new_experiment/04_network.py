"""BabyGPT Step 04: a network.

Step 03:
    many inputs -> one layer -> many outputs

Step 04 connects multiple layers.

The output of the first layer becomes the input of the next layer:

    inputs -> layer 1 -> layer 2 -> outputs

We keep everything in plain Python so the data flow stays visible.
"""


def neuron(inputs, weights, bias):
    # One neuron combines many inputs into one output.
    total = sum(input_value * weight for input_value, weight in zip(inputs, weights))
    return total + bias


def layer(inputs, neurons):
    # A layer is many neurons processing the same inputs.
    return [
        neuron(inputs, weights, bias)
        for weights, bias in neurons
    ]


def network(inputs, layers):
    # Each layer receives the output of the previous layer.
    output = inputs

    for current_layer in layers:
        output = layer(output, current_layer)

    return output


inputs = [2, 3]

layers = [
    [
        ([0.5, 1], 1),
        ([2, -1], 0),
    ],
    [
        ([1, 2], 0),
        ([-1, 1], 1),
    ],
]

outputs = network(inputs, layers)

print("inputs :", inputs)
print("outputs:", outputs)
