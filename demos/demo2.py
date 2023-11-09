import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(f"{project_root}/src")

from optimizers import NAdam
import numpy as np
from main import NN
from layers import Input, Flatten, Conv2D, Dropout, Dense



def load_data() -> tuple:
    from keras.datasets import mnist
    from keras.utils import to_categorical

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = np.reshape(X_train, (60000, 28, 28, 1)) / 255
    X_test = np.reshape(X_test, (10000, 28, 28, 1)) / 255

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (X_train, X_test, y_train, y_test)


X_train, X_test, y_train, y_test = load_data()

np.random.seed(1337)

print("\033c", end='')

model = NN("NN for MNIST")
model.add(Input((28, 28, 1)))
model.add(Conv2D(16, strides=(2, 2), name="Conv 1"))
model.add(Flatten())
model.add(Dropout(100, "relu", 0.5))
model.add(Dense(50, "relu"))
model.add(Dense(10, "softmax"))

optimizer = NAdam(adjust_biases_shape=True)

model.compile("bce", optimizer=optimizer,
              weight_initaliziton="he", metrics="accuracy")
model.summary()


model.train(X_train, y_train, 100, verbose=2)
