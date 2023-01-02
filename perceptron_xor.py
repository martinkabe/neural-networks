import numpy as np

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
outputs = np.array([0,1,1,0])
weights = np.array([0.0,0.0])
alpha = 0.1 # learning rate
epochs = 10

def step(sum_inpt):
    if sum_inpt >= 1:
        return 1
    return 0

def calculate_output(inpts):
    return step(inpts.dot(weights))

def train():
    for e in range(epochs):
        total_error = 0
        print(f'Epoch #{e}:')
        for i in range(len(inputs)):
            prediction = calculate_output(inputs[i])
            error = abs(prediction - outputs[outputs[i]])
            total_error += error
            if total_error > 0:
                # adjust weights
                for w in range(len(weights)):
                    weights[w] = weights[w] + (alpha * inputs[i][w] * error)
                    print(f'Weight updated: {weights[w]}')
        print(f'Total Error = {total_error}')
        if total_error == 0:
            print(f'\nNN trained in #{e} epochs.\n')
            break


if __name__ == '__main__':
    train()
