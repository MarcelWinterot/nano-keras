from nano_keras.optimizers import NAdam
import numpy as np
from nano_keras.models import NN
from nano_keras.layers import Input, Flatten, Conv2D, Dropout


def load_data() -> tuple:
    # Note that we use keras to only downlaod the dataset and modify it's values as
    # I'm focusing on creating NNs and not playing with data
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
model.add(Conv2D(32, (2, 2), (2, 2), name="Conv 1"))
model.add(Conv2D(64, (2, 2), (2, 2), name='Conv 2'))
model.add(Flatten())
model.add(Dropout(10, "relu", 0.5))

optimizer = NAdam(adjust_biases_shape=True)

model.compile("bce", optimizer=optimizer,
              weight_initaliziton="he", metrics="accuracy", weight_data_type=np.float32)
model.summary()


model.train(X_train, y_train, 100, verbose=2)
