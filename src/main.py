import numpy as np
import matplotlib.pyplot as plt
from activations import *
from losses import *
from optimizers import *
from regulizers import *
from typing import Union


def print_progress(epoch: int, totalEpochs: int, loss: float) -> None:
    barLength = 30
    progress = int(barLength * epoch / totalEpochs)

    progress_bar = "[" + "=" * progress + \
        ">" + "." * (barLength - progress) + "]"
    print(
        f"\r{epoch}/{totalEpochs} {progress_bar} - loss: {loss:.12f}", end='')



class Layer:
    def __init__(self, units: int, activation: Union[Activation, str], regulizer=Union[Regularizer, None], name: str = "Layer") -> None:
        __activations__ = {'sigmoid': Sigmoid(), 'tanh': Tanh(
        ), 'relu': ReLU(), 'leaky_relu': LeakyReLU(), 'elu': ELU()}
        self.units = units
        self.name = name
        self.weights = np.array([])
        self.biases = np.random.randn(units)
        self.activation = __activations__[activation] if type(activation) == str else activation
        self.regulizer = regulizer

    def summary(self) -> None:
        print(
            f"{self.name}: {self.units} units")

    def feed_forward(self, x: np.ndarray) -> tuple:
        weighted_sum = np.dot(x, self.weights) + self.biases
        output = self.activation.compute_loss(weighted_sum)
        return output, weighted_sum

    def backpropagate(self, loss: np.ndarray, outputs: np.ndarray, inputs: np.ndarray, optimizer: Optimizer) -> np.ndarray:
        if type(self.regulizer) == Optimizer:
            loss = self.regulizer.compute_loss(loss, self.weights, self.biases)
        delta = np.average([(loss * self.activation.compute_derivative(outputs[i]))
                            for i in range(len(outputs))])
        weights_gradients = np.outer(inputs, delta)
        weights, biases = optimizer.apply_gradients(weights_gradients, np.array(
            [delta], dtype=float), self.weights, self.biases)
        self.weights = weights
        self.biases = biases
        loss = np.dot(delta, self.weights.T)
        return loss


class NN:
    def __init__(self, name: str = "NN"):
        self.name = name
        self.layers = []

    def add(self, layer: Layer):
        self.layers.append(layer)

    def summary(self, line_length: int = 50) -> None:
        print(f"{self.name}:\n{'='*line_length}")
        for layer in self.layers:
            layer.summary()
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
        outputs, inputs = [], []
        for layer in self.layers[1:]:
            inputs.append(output)
            output, weighted_sum = layer.feed_forward(output)
            outputs.append([output, weighted_sum])
        return output, outputs, inputs
# 
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int) -> np.ndarray:
        losses = np.ndarray((epochs))
        for epoch in range(epochs):
            for i in range(len(X)):
                yPred, outputs, inputs = self.feed_forward(X[i])
                loss = self.loss_function.compute_derivative(y[i], yPred)
                for i in range(len(self.layers)-1, 0, -1):
                    loss = self.layers[i].backpropagate(
                        loss, outputs[i-1], inputs[i-1], self.optimizer)
            loss = self.evaluate(X, y)
            losses[epoch] = loss
            print_progress(epoch+1, epochs, loss)
        return losses

    def evaluate(self, X: np.ndarray, y: np.ndarray, showPreds: bool = False) -> float:
        yPreds = np.ndarray((X.shape[0], self.layers[-1].units))

        for i, x in enumerate(X):
            result, _, _ = self.feed_forward(x)
            yPreds[i] = result

        if showPreds:
            print(yPreds)
        return self.loss_function.compute_loss(y, yPreds)


if __name__ == "__main__":
    np.random.seed(1337)
    Network = NN()

    # regulizaer = L1L2(1e-4, 1e-5)

    Network.add(Layer(2, "sigmoid", name="Input"))
    Network.add(Layer(2, "leaky_relu", name="Hidden"))
    Network.add(Layer(1, "sigmoid", name="Output"))

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
