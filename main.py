import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])
training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_weight = 2 * np.random.random((3,1)) - 1

print("Случайные веса:")
print(synaptic_weight)

# Метод обратного распростронения ( выбираем его , посколько нам важно понимание самого процесса обучения)
for i in range(500000):
    input_layer = training_inputs
    outputs = sigmoid( np.dot(input_layer,  synaptic_weight))

    err = training_outputs - outputs
    adjustments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))

    synaptic_weight += adjustments


print("Веса после обучения нейронной сети:")
print( synaptic_weight)
print("Результат после обучения нейронной сети:")
print(outputs)