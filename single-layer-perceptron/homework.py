import numpy as np
from sklearn.preprocessing import MinMaxScaler

inputs = np.array([[18,2], [20,3], [21, 4],  
                   [35,15], [36,16], [38, 18]])

# Transform features by scaling each feature to a given range.
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
scaler = MinMaxScaler()
inputs_transformed = scaler.fit_transform(inputs)
print(f'Transformed inputs are:\n{inputs_transformed}')

outputs = np.array([0, 0, 0, 1, 1, 1])
weights = np.array([0.0, 0.0])
alpha = 0.1 # learning rate
epochs = 10

def step(sum_inpt):
    if sum_inpt >= 1:
        return 1
    return 0

def calculate_output(inpts):
    return step(inpts.dot(weights))

def test_function(inpts):
    for inpt in inpts:
        print(f'Output for [{inpt}] is \t{calculate_output(inpt)}')

def train():
    for epoch in range(epochs):
        print(f'\nEpoch #{epoch}')
        total_error = 0
        for i in range(len(outputs)):
            predicted = calculate_output(inputs_transformed[i])
            error = abs(outputs[i] - predicted)
            total_error += error
            if error > 0:
                # update weights
                for j in range(len(weights)):
                    weights[j] += weights[j] + (alpha * inputs_transformed[i][j] * error)
                    print(f'New weights: {weights}')
        print(f'Total error is {total_error}')
        if total_error == 0:
            break

train()

print('Test the NN:\n')
test_inputs = np.array([[17,5], [25,8],  
                        [45,10], [31,20]])
test_inputs_transformed = scaler.transform(test_inputs)
print(f'Test inputs transformed are:\n{test_inputs_transformed}')
print('\n---------------------Results---------------------\n')
test_function(test_inputs_transformed)
