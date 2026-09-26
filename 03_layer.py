"""BabyGPT Step 03: a layer.

Step 02:
    one input -> many neurons -> many outputs

Step 03 adds the next missing piece:
    many inputs -> one neuron -> one output

A neuron now combines several inputs:

    output = input1 * weight1
           + input2 * weight2
           + bias

A layer is then several of these neurons working on the same inputs.
"""

from typing import List


def neuron(inputs: List[float], weights: List[float], bias: float):
    # Each input is multiplied by its matching weight.
    # The neuron adds all contributions, then shifts the result with bias.
    total = sum(input_value * weight for input_value, weight in zip(inputs, weights))
    return total + bias


def layer(inputs: List[float], neurons):
    # Every neuron receives the same set of inputs,
    # but each neuron has different weights and bias.
    return [
        neuron(inputs, weights, bias)
        for weights, bias in neurons
    ]


inputs = [2, 3]

neurons = [
    ([0.5, 1], 1),    # neuron 1
    ([2, -1], 0),     # neuron 2
    ([-1, 0.5], 2),   # neuron 3
]

outputs = layer(inputs, neurons)

print("inputs :", inputs)
print("outputs:", outputs)
