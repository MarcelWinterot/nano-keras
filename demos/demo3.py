"""
For now this demo is used for testing what works, and showcasing progress on the LSTM
In the future it will have RNN implemented
"""

from nano_keras.models import NN
from nano_keras.layers import LSTM, Input, Dense
import numpy as np

model = NN()

model.add(Input((12, 30)))
model.add(LSTM(10, name="Lstm 1"))
model.add(LSTM(5, return_sequences=False, name="Lstm 2"))
model.add(Dense(10, "sigmoid"))

model.compile()

model.summary()

x = np.random.randn(12, 30)

output = model.feed_forward(x).shape
output = model.feed_forward(x).shape

print(output)
