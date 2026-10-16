"""BabyGPT Step 02: many neurons.

One neuron gives us one output.
The next problem is simple:

    What if we need several outputs from the same input?

We solve that by using several neurons.
Each neuron has its own weight and bias.
"""

from typing import List, Tuple


def neuron(input_value, weight, bias):
    # One neuron still does the same simple calculation.
    return input_value * weight + bias


def many_neurons(input_value, neurons: List[Tuple[float, float]]):
    # Each neuron receives the same input,
    # but has its own weight and bias.
    return [
        neuron(input_value, weight, bias)
        for weight, bias in neurons
    ]


input_value = 2

neurons = [
    (0.5, 1),   # neuron 1
    (2, 0),     # neuron 2
    (-1, 3),    # neuron 3
]

outputs = many_neurons(input_value, neurons)

print("input :", input_value)
print("outputs:", outputs)
