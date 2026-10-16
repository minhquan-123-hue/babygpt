"""BabyGPT Step 01: one neuron.

A neuron is just this calculation:

    output = input * weight + bias

We use plain Python on purpose so the data flow is visible.
"""


def neuron(input_value, weight, bias):
    # Weight controls how strongly the input affects the output.
    # Bias shifts the final result.
    return input_value * weight + bias


input_value = 2
weight = 0.5
bias = 1

output = neuron(input_value, weight, bias)

print("input :", input_value)
print("weight:", weight)
print("bias  :", bias)
print("output:", output)
