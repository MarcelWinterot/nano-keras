"""
For now this demo is used for testing what works, and showcasing progress on LSTM and other RNN layers
In the future it will have RNN implemented
"""
from nano_keras.models import NN
from nano_keras.layers import LSTM, Embedding, GRU
from nano_keras.optimizers import SGD
from nano_keras.losses import MSE
import numpy as np

model = NN()

model.add(Embedding(10, 20, input_length=14))
model.add(LSTM(8, name="LSTM 1"))
model.add(GRU(6, name="GRU 1"))
model.add(GRU(20, name="GRU 2"))
model.add(LSTM(4, return_sequences=False, name="LSTM 2"))

model.compile(optimizer="sgd")
model.summary()

x = np.random.randint(0, 10, size=(14,))
y = np.random.randn(4)

output = model.feed_forward(x)

print(f"Output shape: {output.shape}")

gradient = MSE().compute_derivative(y, output)

opt = SGD()

for layer in model.layers[-1::-1]:
    print(f"\nGradient shape: {gradient.shape}")
    gradient = layer.backpropagate(gradient, [opt, opt])
