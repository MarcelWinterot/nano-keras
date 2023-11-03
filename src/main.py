import sys
import numpy as np
import matplotlib.pyplot as plt
from activations import *
from losses import *
from optimizers import *
from regulizers import *
from layers import *
from callbacks import *

"""
TODO Overall:
1. Add backpropagation for Conv1D and Conv2D layers
2. Fix the loss functions.
"""


def print_progress(epoch: int, total_epochs: int, loss: float, accuracy: float = None, batch: int = None, total_batches: int = None, val_loss: float = None, val_accuracy: float = None) -> None:
    """Function that prints out current training progress

    Args:
        epoch (int): current epoch
        total_epochs (int): total number of epochs
        loss (float): loss calculated by NN.evaluate function
        accuracy (float, optional): accuracy of the model during training. Default to None.
        batch (int, optional): current batch. Will be set if verbose is set to 2 in the NN.train function. Defaults to None.
        total_batches (int, optional): total number of batches. Will be set if verbose is set to 2 in the NN.train function. Defaults to None.
    """
    bar_length = 30

    if batch is not None and total_batches is not None:
        progress = int(bar_length * batch / total_batches)
        progress_bar = f"[{'=' * progress}>{'.' * (bar_length - progress)}]"
        progress_info = f"{epoch}/{total_epochs}: {progress_bar} - batch: {batch}/{total_batches} - loss: {loss:.8f}"

    else:
        progress = int(bar_length * epoch / total_epochs)
        progress_bar = f"[{'=' * progress}>{'.' * (bar_length - progress)}]"
        progress_info = f"{epoch}/{total_epochs} {progress_bar} - loss: {loss:.8f}"

    if accuracy is not None:
        progress_info += f" - accuracy: {accuracy:.2f}"

    if val_loss is not None:
        progress_info += f" - val_loss: {val_loss:.8f} - val_accuracy: {val_accuracy:.2f}"

    print(f"\r{progress_info}", end='')


class NN:
    def __init__(self, name: str = "NN"):
        """NN init function. Simmilar to the keras.models.Sequential class. Simply add layers using NN.add(Layer)

        Args:
            name (str, optional): Name of the NN. Defaults to "NN".
        """
        self.name = name
        self.layers = []
        self.loss = 1e50
        self.accuracy = 0
        self.val_loss = None
        self.val_accuracy = None
        self.layers_without_units = [
            Flatten, Reshape, MaxPooling1D, MaxPooling2D, Conv1D, Conv2D]
        self.trainable_layers = [Dense, Dropout, Conv1D, Conv2D]

    def add(self, layer: Layer):
        """Adds a custom layer to the NN.

        Args:
            layer (Layer): Custom layer you want to add. You can find them in the layers.py. You have to add the already initialized layer and not the Class itself.
        """
        self.layers.append(layer)

    def summary(self, line_length: int = 65) -> None:
        """Prints out the NN's information.

        Args:
            line_length (int, optional): Sets how long the string will be. Defaults to 50.
        """
        print(f"Model: {self.name}\n{'_'*line_length}")
        print(
            f"Layer (type)                Output Shape              Param #\n{'='*line_length}")
        params = []
        totalParams = 0
        for layer in self.layers:
            print(layer)
            if any(isinstance(layer, trainable_layer) for trainable_layer in self.trainable_layers):
                totalParams += layer.weights.size + layer.biases.size
                params.append([layer.weights, layer.biases])
        print(f"{'='*line_length}")
        print(
            f"Total params: {totalParams} ({sys.getsizeof(params)} Bytes)")
        print(f"{'_'*line_length}")

    def generate_weights(self) -> None:
        """Support function used in the compile function to generate model's weights.
        """
        for i in range(1, len(self.layers)):
            if not any(isinstance(self.layers[i], layer) for layer in self.layers_without_units):
                previousUnits = self.layers[i-1].output_shape(self.layers, i-1)
                try:
                    weights = np.random.randn(
                        previousUnits, self.layers[i].units)
                except Exception as e:
                    print(
                        f"Exception encountered when creating weights: {e}\nChange your achitecture and try again. If you think it's an error post an issue on github")
                    sys.exit()
                self.layers[i].weights = weights

    def compile(self, loss_function: Loss | str = "mse", optimizer: Optimizer | str = "adam", metrics: str = "") -> None:
        """Function you should call before starting training the model, as we generate the weights in here, set the loss function and optimizer.

        Args:
            loss_function (Loss | str): Loss function the model should use. You can pass either the name of it as a str or intialized class. Defaults to "mse".
            optimizer (Optimizer | str): Optimizer the model should use when updating it's params. You can pass either the name of it as a str or initalized class. Defaults to "adam"
            metrics (str, optional): Paramter that specifies what metrics should the model use. Possible metrics are: accuracy. Defaults to "".
        """
        _loss_functions = {
            "mae": MAE(), "mse": MSE(), "bce": BCE(), "cce": CCE(), "hinge": Hinge(), "huber": Huber()}
        _optimizers = {"adam": Adam(), "sgd": SGD(), "adagrad": Adagrad(
        ), "adadelta": Adadelta(), "rmsprop": RMSProp(), "nadam": NAdam()}
        self.generate_weights()
        self.loss_function = _loss_functions[loss_function] if type(
            loss_function) == str else loss_function
        self.optimizer = _optimizers[optimizer] if type(
            optimizer) == str else optimizer
        self.metrics = metrics

    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        """Feed forward for the whole model

        Args:
            x (np.ndarray): x dataset

        Returns:
            np.ndarray: output of the model 
        """
        output = x
        # We skip the first layer as it's the input layer
        for layer in self.layers[1:]:
            output = layer(output)
        return output

    def backpropagate(self, X: np.ndarray, y: np.ndarray, verbose: int = 2, epoch: int = 1, total_epochs: int = 100) -> None:
        """Backpropgate function to make the train function cleaner and better for future expansion

        Args:
            X (np.ndarray): X dataset
            y (np.ndarray): y dataset
        """
        length_of_x = len(X)
        for i in range(length_of_x):
            yPred = self.feed_forward(X[i])
            loss = self.loss_function.compute_derivative(y[i], yPred)
            # We skip over the input layer, as it doesn't have any parameters to update
            for layer in self.layers[-1:0:-1]:
                loss = layer.backpropagate(loss, self.optimizer)
            if verbose == 2:
                loss, accuracy = self.evaluate(X, y)
                print_progress(epoch+1, total_epochs, loss,
                               accuracy, i+1, length_of_x, self.val_loss, self.val_accuracy)

    def _handle_callbacks(self, result, callbacks: EarlyStopping | None) -> None | np.ndarray:
        """Support function to make the code cleaner for handling the callbacks

        Args:
            callbacks (EarlyStopping | None): Callbacks used by the model. If it isn't set it's None

        Returns:
            None |np.ndarray: Either None if the training continues
        """
        if result is not None:
            if callbacks.restore_best_weights:
                for i, layer in enumerate(self.layers):
                    if result[0][i].size > 0:
                        layer.weights = result[0][i]
                        layer.biases = result[1][i]
            return 1
        return 0

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, callbacks: EarlyStopping | None = None, verbose: int = 1, validation_data: tuple[np.ndarray, np.ndarray] = None) -> np.ndarray | tuple:
        """Function to train the model. Remember to call the NN.compile() before calling this function as it won't work. \n
        Currently the code updates the weights after each parameter instead of working in batches but I will add that in the future

        Args:
            X (np.ndarray): X dataset
            y (np.ndarray): y dataset
            epochs (int): number of iterations a model should do during training
            callbacks (EarlyStopping | None, optional): One of the callbacks implemented in callbacks.py although currently there's only early stopping in there. Defaults to None.
            verbose (int, optional): Parameter to control what the model prints out during training. 0 - nothing, 1 - only epoch/epochs, 2 - all the useful information. Defaults to 1.
            validation_data (tuple, optional): Validation data a model should use to check the validation loss and accuracy. It should be a tuple of X and y. Default to None.

        Returns:
            np.ndarray | tuple: Either the losses model's had during training or both the losses and val_losses if validation_data is set.
        """
        losses = np.ndarray((epochs))
        val_losses = np.ndarray((epochs))
        for epoch in range(epochs):
            self.backpropagate(X, y, verbose, epoch, epochs)
            self.loss, self.accuracy = self.evaluate(X, y)

            if validation_data is not None:
                self.val_loss, self.val_accuracy = self.evaluate(
                    validation_data[0], validation_data[1])

                val_losses[epoch] = self.val_loss

                self._metrics = {"loss": self.loss, "accuracy": self.accuracy,
                                 "val_loss": self.val_loss, "val_accuracy": self.val_accuracy}
            else:
                self._metrics = {"loss": self.loss,
                                 "accuracy": self.accuracy, }

            result = callbacks.watch(
                self._metrics[callbacks.monitor], self.layers) if callbacks is not None else None

            if self._handle_callbacks(result, callbacks) == 1:
                break

            losses[epoch] = self.loss
            if verbose == 1:
                print_progress(epoch+1, epochs, self.loss, self.accuracy,
                               val_loss=self.val_loss, val_accuracy=self.val_accuracy)
        if validation_data is not None:
            return losses, val_losses
        return losses

    def evaluate(self, X: np.ndarray, y: np.ndarray, show_preds: bool = False, min_accuracy_error: float = 0.3) -> float:
        """Model's evaluate function, which returns the loss calculated by the models loss function.

        Args:
            X (np.ndarray): X dataset
            y (np.ndarray): y dataset
            show_preds (bool, optional): If set to True the predictions will be printed out. Defaults to False.
            min_accuracy_error (float, optional): Sets the difference between yTrue and yPred in order for it to be counted as a correct prediction. Defaults to 0.3.

        Returns:
            tuple: loss, accuracy. Accuracy is None if the metrics in NN.compile() isn't set to accuracy 
        """
        yPreds = np.ndarray((X.shape[0], self.layers[-1].units))

        for i, x in enumerate(X):
            yPreds[i] = self.feed_forward(x)

        if show_preds:
            print(yPreds)

        accuracy = None
        if self.metrics == "accuracy":
            accuracy = np.average(np.abs(y - yPreds) < min_accuracy_error)

        return self.loss_function.compute_loss(y, yPreds), accuracy

    def save(self, file_path: str) -> None:
        """Function to save the models params into a file

        Args:
            file_path (str): File path where to save the model. Don't put the file extension as it is alredy handled by numpy.
            For example if you want to save the model at './saved_model' put that as the file_path, and numpy will add the extension
        """
        array_to_save = []
        for layer in self.layers:
            if any(isinstance(layer, layer_without_units) for layer_without_units in self.layers_without_units):
                array_to_save.append([])
                continue
            array_to_save.append([layer.weights, layer.biases])

        array_to_save = np.array(array_to_save, dtype=object)

        np.save(file_path, array_to_save)

        print(f"Saved model at: {file_path}.npy")
        return

    def load(self, file_path: str) -> None:
        """Function to load the models params

        Args:
            file_path (str): File path to the saved weights and biases. You have to also specify the file extension.
            For example if the path to the file looks like this: './saved_model.npy' you have to put that path to the file.
        """
        array = np.load(file_path, allow_pickle=True)
        for i in range(len(array)):
            if len(array[i]) == 0:
                # We've encountered a layer without params so we continue
                continue
            self.layers[i].weights = array[i][0]
            self.layers[i].biases = array[i][1]

        return


if __name__ == "__main__":
    np.random.seed(1337)
    model = NN()

    model.add(Dense(2, name="input"))
    model.add(Dropout(2, "relu", name="hidden"))
    model.add(Dense(1, "sigmoid", name="output"))

    optimizer = Adam(0.2)
    loss = "mse"

    model.compile(loss, optimizer, metrics="accuracy")
    model.summary()

    # AND
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([[1], [0], [0], [1]])

    print("\n\n STARTING TRAINING \n\n")

    losses, val_losses = model.train(
        X, y, 2500, validation_data=(X, y), verbose=1)

    print("\n\n TRAINING FINISHED \n\n")

    loss, accuracy = model.evaluate(X, y, show_preds=True)

    model.save("./array")

    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over time")
    plt.show()
