"""BabyGPT Step 07: learning.

Step 06 could measure error.

Now we need the model to change its weights so that
the next prediction can become better.

For this first learning step, we use a very simple rule:

    if prediction is too low:
        increase the weight

    if prediction is too high:
        decrease the weight

This is intentionally simple.
It is not backpropagation yet.

The goal is to understand the core idea:

    predict -> measure error -> change weight -> predict again
"""


def neuron(input_value, weight, bias):
    return input_value * weight + bias


def predict(input_value, weight, bias):
    return neuron(input_value, weight, bias)


def loss(prediction, target):
    return (prediction - target) ** 2


def learn(input_value, weight, bias, target, learning_rate):
    prediction = predict(input_value, weight, bias)
    error = target - prediction

    # Move the weight in the direction that reduces the error.
    weight += learning_rate * error

    return weight


input_value = 2
weight = 0.5
bias = 0
target = 4
learning_rate = 0.1

print("=== BEFORE LEARNING ===")

prediction = predict(input_value, weight, bias)
error = loss(prediction, target)

print("weight    :", weight)
print("prediction:", prediction)
print("target    :", target)
print("error     :", error)


weight = learn(
    input_value,
    weight,
    bias,
    target,
    learning_rate,
)

print("\n=== AFTER LEARNING ===")

prediction = predict(input_value, weight, bias)
error = loss(prediction, target)

print("weight    :", weight)
print("prediction:", prediction)
print("target    :", target)
print("error     :", error)
