import numpy as np

def sigmoid(sum):
    return 1 / (1 + np.exp(-sum))

def sigmoid_derivative(sigmoid):
    return sigmoid * (1 - sigmoid)

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])
weights0 = np.array([
    [-0.424, -0.740, -0.961],
    [0.358, -0.577, -0.469]
])
weights1 = np.array([[-0.017], [-0.893], [0.148]])
epochs = 100

input_layer = inputs

sum_synapse0 = np.dot(input_layer, weights0)
hiden_layer = sigmoid(sum_synapse0)
# print(hiden_layer)

sum_synapse1 = np.dot(hiden_layer, weights1)
output_layer = sigmoid(sum_synapse1)

error_output_layer = outputs - output_layer
average = np.mean(abs(error_output_layer))

## delta_output = error * sigmoid_derivative
derivative_output = sigmoid_derivative(output_layer)
# print(derivative_output)
delta_output = error_output_layer * derivative_output

## delta_hidden = sigmoid_derivative * weight * delta_output
weights1_T = weights1.T
delta_output_weights1T = delta_output.dot(weights1_T)
# print(delta_output_weights1T)
delta_hidden_layer = sigmoid_derivative(hiden_layer) * delta_output_weights1T
print(delta_hidden_layer)


for epoch in range(epochs):
    pass
