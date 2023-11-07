from layers import Input, Flatten, Conv2D, Dense
from main import NN
import numpy as np
from keras.datasets import mnist  # We are using keras to only download the dataset
from optimizers import NAdam

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.reshape(X_train, (60000, 28, 28, 1)) / 255
X_test = np.reshape(X_test, (10000, 28, 28, 1)) / 255

print("\033c", end='')

model = NN()
model.add(Input((28, 28, 1)))  # Working
# model.add(Conv2D(16, name="Conv 1"))  # NOT WORKING
model.add(Conv2D(16, name="Conv21"))  # Working
model.add(Flatten())  # Working
model.add(Dense(125, "relu"))  # Working
model.add(Dense(50, "relu"))  # Working
model.add(Dense(25, "relu"))  # Working
model.add(Dense(10, "softmax"))  # Working

optimizer = NAdam(adjust_biases_shape=True)

model.compile(optimizer=optimizer)
model.summary()


model.train(X_train, y_train, 100, verbose=2)
