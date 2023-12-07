"""
For now this demo is used for testing what works, and showcasing progress on LSTM and other RNN layers
In the future it will have RNN implemented
"""
from nano_keras.models import NN
from nano_keras.layers import LSTM, Input
from nano_keras.losses import MSE
from nano_keras.optimizers import SGD
import numpy as np

model = NN()

model.add(Input((12, 30)))
model.add(LSTM(10, name="Lstm 1"))
model.add(LSTM(30, name="Lstm 3"))
model.add(LSTM(5, return_sequences=False, name="Lstm 2"))

model.compile()
model.summary()

x = np.random.randn(12, 30)
y = np.random.randn(5)

output = model.feed_forward(x)

loss = MSE()
opt = SGD()

gradient = loss.compute_derivative(y, output)

for layer in model.layers[-1::-1]:
    print(f"Gradient shape: {gradient.shape}")
    gradient = layer.backpropagate(gradient, opt)
