"""
For now this demo is used for testing what works, and showcasing progress on LSTM and other RNN layers
In the future it will have RNN implemented
"""
from nano_keras.models import NN
from nano_keras.layers import LSTM, Embedding, GRU, MultiHeadAttention
from nano_keras.optimizers import SGD
from nano_keras.losses import MSE
import numpy as np

# Initalizing the model
model = NN()
model.add(Embedding(10, 20, input_length=14))
model.add(LSTM(8, name="LSTM 1"))
model.add(GRU(6, name="GRU 1"))
model.add(GRU(20, name="GRU 2"))
model.add(MultiHeadAttention(4, 8, name="MHA 1"))
model.add(LSTM(4, return_sequences=False, name="LSTM 2"))

model.compile(optimizer="sgd")
model.summary()

# Creating random data for training
x = np.random.randint(0, 10, size=(14,))
y = np.random.randn(4,)

for layer in model.layers:
    layer.set_batch_size(1, model.layers, model.layers.index(layer))

# Feed forward
outputs = model.feed_forward(x, is_training=True)

print(f"Output shape: {outputs.shape}")

# Backpropagation
gradient = MSE().compute_derivative(y, outputs)

opt = SGD()

for layer in model.layers[-1::-1]:
    print(f"\nGradient shape: {gradient.shape}")
    gradient = layer.backpropagate(gradient, [opt, opt])
