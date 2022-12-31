import numpy as np

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
outputs = np.array([0,0,0,1])
weights = np.array([0.0,0.1])
alpha = 0.1 # learning rate

def step_fnc(sum_fnc):
    if sum_fnc >= 1:
        return 1
    return 0

def calculate_output(instance):
    s = instance.dot(weights)
    return step_fnc(s)

def train():
    total_error = 1
    while total_error != 0:
        total_error = 0
        for i in range(len(outputs)):
            prediction = calculate_output(inputs[i])
            error = abs(outputs[i] - prediction)
            total_error += error
            if total_error > 0:
                for j in range(len(weights)):
                    weights[j] = weights[j] + (alpha * inputs[i][j] * error)
                    print(f'Weight updated: {weights[j]}')
        print(f'Total error: {total_error}')

train()

# Classification
