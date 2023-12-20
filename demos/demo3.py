"""
For now this demo is used for testing what works, and showcasing progress on LSTM and other RNN layers
In the future it will have RNN implemented
"""
from nano_keras.models import NN
from nano_keras.layers import LSTM, Input, GRU
from nano_keras.losses import MSE
from nano_keras.optimizers import SGD
import numpy as np

model = NN()

model.add(Input((5, 14)))
model.add(LSTM(8, name="LSTM 1"))
model.add(GRU(6, name="LSTM 2"))
model.add(GRU(20, name="LSTM 3"))
model.add(LSTM(4, return_sequences=False, name="LSTM 4"))

model.compile()
model.summary()

x = np.random.randn(5, 14)
y = np.random.randn(4)

output = model.feed_forward(x)

loss = MSE()
opt = SGD()

gradient = loss.compute_derivative(y, output)

for layer in model.layers[-1::-1]:
    print(f"\nGradient shape: {gradient.shape}")
    gradient = layer.backpropagate(gradient, [opt, opt])
