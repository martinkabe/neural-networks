import numpy as np

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])
weights0 = np.array([
    [-0.424, -0.740, -0.961],
    [0.358, -0.577, -0.469]
])
weights1 = np.array([[-0.017], [-0.893], [0.148]])
epochs = 100


def sigmoid(sum):
    return 1 / (1 + np.exp(-sum))


input_layer = inputs

sum_synapse0 = np.dot(input_layer, weights0)
hiden_layer = sigmoid(sum_synapse0)
# print(hiden_layer)

sum_synapse1 = np.dot(hiden_layer, weights1)
output_layer = sigmoid(sum_synapse1)
print(output_layer)

for epoch in range(epochs):
    pass
