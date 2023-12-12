import numpy as np
from nano_keras.optimizers import Adam
from nano_keras.models import NN
from nano_keras.layers import Input, Flatten, Conv2D, MaxPool2D, Dense


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

    return X_train, X_test, y_train, y_test


print(f"Started downloading the data. It might take a minute or two")

X_train, X_test, y_train, y_test = load_data()


np.random.seed(1337)

print("\033c", end='')

model = NN(name="NN for MNIST")
model.add(Input((28, 28, 1)))
model.add(Conv2D(32, (3, 3), (2, 2), name="Conv 1"))
# model.add(MaxPool2D())
model.add(Conv2D(64, (3, 3), (2, 2), name='Conv 2'))
# model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(10, "relu", "he_normal", name='Dropout'))

optimizer = Adam(adjust_biases_shape=True, learning_rate=0.01)

model.compile("mse", optimizer=optimizer, metrics="accuracy",
              weight_data_type=np.float32)

model.summary()

model.train(X_train, y_train, 25, 100, verbose=2)
