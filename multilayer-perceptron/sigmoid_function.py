import numpy as np


def sigmoid(sum):
    return 1 / (1 + np.exp(-sum))


# test sigmoid function
for num in np.arange(-30.5, 30.0, 0.5):
    print(f'Sigmoid ({num}) = {sigmoid(num)}')
