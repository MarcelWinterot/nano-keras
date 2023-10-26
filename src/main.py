import numpy as np
import matplotlib.pyplot as plt
from activations import *
from losses import *
from optimizers import *
from regulizers import *
from layers import *


def print_progress(epoch: int, totalEpochs: int, loss: float) -> None:
    barLength = 30
    progress = int(barLength * epoch / totalEpochs)

    progress_bar = "[" + "=" * progress + \
        ">" + "." * (barLength - progress) + "]"
    print(
        f"\r{epoch}/{totalEpochs} {progress_bar} - loss: {loss:.12f}", end='')


class NN:
    def __init__(self, name: str = "NN"):
        self.name = name
        self.layers = []

    def add(self, layer: Layer):
        self.layers.append(layer)

    def summary(self, line_length: int = 50) -> None:
        print(f"{self.name}:\n{'='*line_length}")
        for layer in self.layers:
            print(layer)
        print('='*line_length)

    def generate_weights(self) -> None:
        for i in range(1, len(self.layers)):
            prev_units = self.layers[i-1].units
            curr_units = self.layers[i].units
            weights = np.random.randn(prev_units, curr_units)
            self.layers[i].weights = weights

    def compile(self, loss_function: Loss, optimizer: Optimizer) -> None:
        self.generate_weights()
        self.loss_function = loss_function
        self.optimizer = optimizer

    def feed_forward(self, x: np.ndarray) -> tuple:
        output = x
        for layer in self.layers[1:]:
            output = layer.feed_forward(output)
        return output

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int) -> np.ndarray:
        losses = np.ndarray((epochs))
        for epoch in range(epochs):
            for i in range(len(X)):
                yPred = self.feed_forward(X[i])
                loss = self.loss_function.compute_derivative(y[i], yPred)
                for i in range(len(self.layers)-1, 0, -1):
                    loss = self.layers[i].backpropagate(loss, self.optimizer)
            loss = self.evaluate(X, y)
            losses[epoch] = loss
            print_progress(epoch+1, epochs, loss)
        return losses

    def evaluate(self, X: np.ndarray, y: np.ndarray, showPreds: bool = False) -> float:
        yPreds = np.ndarray((X.shape[0], self.layers[-1].units))

        for i, x in enumerate(X):
            result = self.feed_forward(x)
            yPreds[i] = result

        if showPreds:
            print(yPreds)
        return self.loss_function.compute_loss(y, yPreds)


if __name__ == "__main__":
    np.random.seed(1337)
    Network = NN()

    # regulizaer = L1L2(1e-4, 1e-5)

    Network.add(Dense(2, "sigmoid"))
    Network.add(Dense(2, "relu"))
    Network.add(Dropout(2, "relu", 0.2))
    Network.add(Dense(1, "sigmoid"))

    optimizer = Adam(learningRate=0.2)
    loss = MSE()

    Network.compile(loss, optimizer)
    Network.summary()

    # AND
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([[0], [0], [0], [1]])

    print("\n\n STARTING TRAINING \n\n")

    losses = Network.train(X, y, 2500)

    print("\n\n TRAINING FINISHED \n\n")

    Network.evaluate(X, y, showPreds=True)

    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over time")
    plt.show()
