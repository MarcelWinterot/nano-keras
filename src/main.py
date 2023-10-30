import numpy as np
import matplotlib.pyplot as plt
from activations import *
from losses import *
from optimizers import *
from regulizers import *
from layers import *
from callbacks import *

"""
TODO 28.10.2023
1. Try to calculate the derivatives of loss functions
2. Learn how do the final layers left to implement work
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
        # self.__metrics__ = {"loss": self.loss, "accuracy": self.accuracy,
        #                     "val_loss": self.val_loss, "val_accuracy": self.val_accuracy}

    def add(self, layer: Layer):
        """Adds a custom layer to the NN.

        Args:
            layer (Layer): Custom layer you want to add. You can find them in the layers.py. You have to add the already initialized layer and not the Class itself.
        """
        self.layers.append(layer)

    def summary(self, line_length: int = 50) -> None:
        """Prints out the NN's information.

        Args:
            line_length (int, optional): Sets how long the string will be. Defaults to 50.
        """
        print(f"{self.name}:\n{'='*line_length}")
        for layer in self.layers:
            print(layer)
        print('='*line_length)

    def generate_weights(self) -> None:
        """Support function used in the compile function to generate model's weights.
        """
        # TODO Make the code cleaner and throw errors when we have more than 2 layers without weights next to each other.
        for i in range(1, len(self.layers)):
            # Kinda primitive but it works, and as the rule says, if something works don't touch it if you don't want to break it
            if self.layers[i].type not in [Flatten, Reshape]:
                prev_units = self.layers[i-1].units if self.layers[i -
                                                                   1].type not in [Flatten, Reshape] else self.layers[i-2].units
                curr_units = self.layers[i].units
                weights = np.random.randn(prev_units, curr_units)
                self.layers[i].weights = weights

    def compile(self, loss_function: Union[Loss, str], optimizer: Optimizer, metrics: str = "") -> None:
        """Function you should call before starting training the model, as we generate the weights in here, set the loss function and optimizer.

        Args:
            loss_function (Union[Loss, str]): Loss function the model should use. You can either pass the name of it as a str or intialized class.
            optimizer (Optimizer): Optimizer the model should use when updating it's params. You should pass the already intialized class not the class itself.
            metrics (str, optional): Paramter that specifies what metrics should the model use. Possible metrics are: accuracy. Defaults to "".
        """
        self.__loss_functions__ = {
            "mae": MAE(), "mse": MSE(), "bce": BCE(), "cce": CCE(), "hinge": Hinge(), "huber": Huber()}
        self.generate_weights()
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metrics = metrics

    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        """Feed forward for the whole model

        Args:
            x (np.ndarray): x dataset

        Returns:
            np.ndarray: output of the model 
        """
        output = x
        for layer in self.layers[1:]:
            output = layer.feed_forward(output)
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
            for j in range(len(self.layers)-1, 0, -1):
                loss = self.layers[j].backpropagate(
                    loss, self.optimizer)
            if verbose == 2:
                loss, accuracy = self.evaluate(X, y)
                print_progress(epoch+1, total_epochs, loss,
                               accuracy, i+1, length_of_x, self.val_loss, self.val_accuracy)

    def __handle_callbacks__(self, result, callbacks: Union[EarlyStopping, None]) -> Union[None, np.ndarray]:
        """Support function to make the code cleaner for handling the callbacks

        Args:
            callbacks (Union[EarlyStopping, None]): Callbacks used by the model. If it isn't set it's None

        Returns:
            Union[None, np.ndarray]: Either None if the training continues
        """
        if result is not None:
            if callbacks.restore_best_weights:
                for i, layer in enumerate(self.layers):
                    if result[0][i].size > 0:
                        layer.weights = result[0][i]
                        layer.biases = result[1][i]
            return 1
        return 0

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, callbacks: Union[EarlyStopping, None] = None, verbose: int = 1, validation_data: tuple[np.ndarray, np.ndarray] = None) -> Union[np.ndarray, tuple]:
        """Function to train the model. Remember to call the NN.compile() before calling this function as it won't work. \n
        Currently the code updates the weights after each parameter instead of working in batches but I will add that in the future

        Args:
            X (np.ndarray): X dataset
            y (np.ndarray): y dataset
            epochs (int): number of iterations a model should do during training
            callbacks (Union[EarlyStopping, None], optional): One of the callbacks implemented in callbacks.py although currently there's only early stopping in there. Defaults to None.
            verbose (int, optional): Parameter to control what the model prints out during training. 0 - nothing, 1 - only epoch/epochs, 2 - all the useful information. Defaults to 1.
            validation_data (tuple, optional): Validation data a model should use to check the validation loss and accuracy. It should be a tuple of X and y. Default to None.

        Returns:
            Union[np.ndarray, tuple]: Either the losses model's had during training or both the losses and val_losses if validation_data is set.
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
                self.metrics_ = {"loss": self.loss, "accuracy": self.accuracy,
                                 "val_loss": self.val_loss, "val_accuracy": self.val_accuracy}
            else:
                self.metrics_ = {"loss": self.loss,
                                 "accuracy": self.accuracy, }

            result = callbacks.watch(
                self.metrics_[callbacks.monitor], self.layers) if callbacks is not None else None

            if self.__handle_callbacks__(result, callbacks) == 1:
                break

            losses[epoch] = self.loss
            if verbose == 1:
                print_progress(epoch+1, epochs, self.loss, self.accuracy,
                               val_loss=self.val_loss, val_accuracy=self.val_accuracy)
        if validation_data is not None:
            return losses, val_losses
        return losses

    def evaluate(self, X: np.ndarray, y: np.ndarray, show_preds: bool = False) -> float:
        """Model's evaluate function, which returns the loss calculated by the models loss function.

        Args:
            X (np.ndarray): X dataset
            y (np.ndarray): y dataset
            show_preds (bool, optional): If set to True the predictions will be printed out. Defaults to False.

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
            accuracy = np.average(np.abs(y - yPreds) < 0.25)

        return self.loss_function.compute_loss(y, yPreds), accuracy

    def save(self, file_path: str) -> None:
        """Function to save the models params into a file

        Args:
            file_path (str): File path where to save the model. Don't put the file extension as it is alredy handled by numpy.
            For example if you want to save the model at './saved_model' put that as the file_path, and numpy will add the extension
        """
        # array_to_save = [[layer.weights, layer.biases]
        #                  for layer in self.layers]
        array_to_save = []
        for layer in self.layers:
            try:
                array_to_save.append([layer.weights, layer.biases])
            except:
                array_to_save.append([])

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
                continue
            self.layers[i].weights = array[i][0]
            self.layers[i].biases = array[i][1]

        return


if __name__ == "__main__":
    np.random.seed(1337)
    model = NN()

    regulizaer = L1L2(1e-4, 1e-5)
    call = EarlyStopping(200, "val_accuracy")

    model.add(Dense(2, "sigmoid"))
    model.add(Flatten())
    model.add(Dense(2, "relu"))
    model.add(Dense(1, "sigmoid"))

    optimizer = Adam(learningRate=0.2)
    loss = MSE()

    model.compile(loss, optimizer, metrics="accuracy")
    model.summary()

    # AND
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([[1], [0], [0], [1]])

    print("\n\n STARTING TRAINING \n\n")

    losses, val_losses = model.train(
        X, y, 2500, validation_data=(X, y))
    # losses = model.train(X, y, 2500)

    print("\n\n TRAINING FINISHED \n\n")

    loss, accuracy = model.evaluate(X, y, show_preds=True)

    model.save("./array")

    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over time")
    plt.show()
