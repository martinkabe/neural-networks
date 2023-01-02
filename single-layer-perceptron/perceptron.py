import numpy as np

# Inputs
inputs = np.array([35, 25])

# Weights
weights = np.array([0.8, 0.1])

# Sum function
def sum_fnc(inputs, weights):
    sum = 0
    for i, w in zip(inputs, weights):
        sum += i * w
    return sum

def sum_numpy(inputs, weights):
    return inputs.dot(weights)

# Step function
def step_fnc(sum):
    if sum >= 1:
        return 1
    return 0

# Final result
s = sum_numpy(inputs, weights)
step = step_fnc(s)
print(step)
