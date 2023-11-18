import numpy as np
from nano_keras.losses import LOSS_FUNCTIONS, Loss
from nano_keras.optimizers import OPTIMIZERS, Optimizer
from nano_keras.layers import LAYERS_WITHOUT_UNITS, TRAINABLE_LAYERS, Layer
from nano_keras.callbacks import EarlyStopping
from copy import deepcopy
from time import time

"""
TODO Overall:
1. Optimize Conv2D layer
Conv2d layers are a lot faster than they were initally, but there's still a lot room for upgrades
The best thing we could do is implenet im2col technique for backpropagation function 

2. Review the code so that there aren't any bugs left

3. Add more demos and update the ones that are already shown
"""


class NN:
    def __init__(self, name: str = "NN"):
        """NN init function. Simmilar to the keras.models.Sequential class. Simply add layers using NN.add(Layer)\n
        Note that the first layer is treated as the input layer so it's recommended to use layers.Input for it as you
        can modify the shape of the input data there

        Args:
            name (str, optional): Name of the model. Defaults to "NN".
        """
        self.name: str = name
        self.layers: list[Layer] = []
        self.loss: float = 1e50
        self.accuracy: float = 0
        self.val_loss: float = None
        self.val_accuracy: float = None

    @staticmethod
    def _convert_size(size: int) -> str:
        """Support function to convert bytes into bigger units so it's more readable

        Args:
            size (int): Size in bytes

        Returns:
            str: Bigger units
        """
        units = ['b', 'kb', 'mb', 'gb']
        unit_index = 0

        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1

        return f"{round(size, 3)} {units[unit_index]}"

    @staticmethod
    def _convert_time(seconds: int) -> str:
        hours, minutes = 0, 0

        if seconds >= 3600:
            hours = seconds // 3600
            seconds -= 3600 * hours

        if seconds > 60:
            minutes = seconds // 60
            seconds -= 60 * minutes

        if hours > 0:
            return f"{hours:.0f}h {minutes:.0f}m {seconds:.0f}s"
        return f"{minutes:.0f}m {seconds:.0f}s"

    def print_progress(self, epoch: int, total_epochs: int, loss: float, accuracy: float = None, batch: int = None, total_batches: int = None, time_taken: float = None) -> None:
        """Function that prints out current training progress

        Args:
            epoch (int): current epoch
            total_epochs (int): total number of epochs
            loss (float): loss calculated by NN.evaluate function
            accuracy (float, optional): accuracy of the model during training. Default to None.
            batch (int, optional): current batch. Will be set if verbose is set to 2 in the NN.train function. If it's None we ignore it during printing. Defaults to None.
            total_batches (int, optional): total number of batches. Will be set if verbose is set to 2 in the NN.train function. If it's None we ignore it during printing. Defaults to None.
            val_loss (float, optional): Validation loss of the model. If it's None we ignore it during printing. Defaults to None.
            val_accuracy (float, optional): Validation accuracy of the model. If it's None we ignore it during printing. Defaults to None.
            time_taken (float, optional): Time that a batch has taken to complete. Note that it is only shown if verbose is set to 2. Defaults to None
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

        if time_taken is not None:
            progress_info += f" - ETA: {self._convert_time(time_taken * (total_batches - batch))}"

        if accuracy is not None:
            if type(accuracy) == np.nan:
                progress_info += f" - accuracy: {0:.3f}"
            else:
                progress_info += f" - accuracy: {accuracy:.3f}"

        if self.val_loss is not None:
            progress_info += f" - val_loss: {self.val_loss:.8f} - val_accuracy: {self.val_accuracy:.3f}"

        print(f"\r{progress_info}", end='')

    def add(self, layer: Layer):
        """Adds a custom layer to the NN.

        Args:
            layer (Layer): Custom layer you want to add. You can find them in the layers.py. You have to add the already initialized layer and not the Class itself.
        """
        self.layers.append(layer)

    def summary(self, line_length: int = 65) -> None:
        """Prints out the NN's information.

        Args:
            line_length (int, optional): Sets how long the string will be. Defaults to 65.
        """
        print(f"Model: {self.name}\n{'_'*line_length}")
        print(
            f"Layer (type)                Output Shape              Param #\n{'='*line_length}")
        paramsWeight = 0
        totalParams = 0
        for layer in self.layers:
            print(layer)
            if any(isinstance(layer, trainable_layer) for trainable_layer in TRAINABLE_LAYERS):
                totalParams += layer.weights.size + layer.biases.size
                paramsWeight += layer.weights.nbytes + layer.biases.nbytes
        print(f"{'='*line_length}")
        print(
            f"Total params: {totalParams} ({self._convert_size(paramsWeight)})")
        print(f"{'_'*line_length}")

    def generate_weights(self, weight_data_type: np.float_) -> None:
        """Support function used in the compile function to generate model's weights.

        Args:
            weight_data_type (np.float_): numpy data type in which the models weights should be stored. Use only np.float_ data types.
        """
        for i in range(1, len(self.layers)):
            if not any(isinstance(self.layers[i], layer) for layer in LAYERS_WITHOUT_UNITS):
                try:
                    self.layers[i].generate_weights(
                        self.layers, i, self.weight_initaliziton, weight_data_type)
                except Exception as e:
                    print(
                        f"Exception encountered when creating weights: {e}\nChange your achitecture and try again. If you think it's an error post an issue on github")
                    exit(1)

    def compile(self, loss_function: Loss | str = "mse", optimizer: Optimizer | str = "adam", metrics: str = "", weight_initaliziton: str = "random", weight_data_type: np.float_ = np.float64) -> None:
        """Function you should call before starting training the model, as we generate the weights in here, set the loss function and optimizer.

        Args:
            loss_function (Loss | str, optional): Loss function the model should use. You can pass either the name of it as a str or intialized class. Defaults to "mse".
            optimizer (Optimizer | str, optional): Optimizer the model should use when updating it's params. You can pass either the name of it as a str or initalized class. Defaults to "adam"
            metrics (str, optional): Paramter that specifies what metrics should the model use. Possible metrics are: accuracy. Defaults to "".
            weight_initaliziton (str, optional): Weights intialization function you want to use for weight intialization. Your options are: random, xavier, he. Defalut to "random"
            weight_data_type (np.float_, optional): Data type you want the models weights to be. Use np.float_ types like np.float32 or np.float64. Defaults to np.float64.
        """
        self.loss_function = LOSS_FUNCTIONS[loss_function] if type(
            loss_function) == str else loss_function

        self.optimizer = OPTIMIZERS[optimizer] if type(
            optimizer) == str else optimizer
        optimizer4d = deepcopy(self.optimizer)
        self.optimizer = [self.optimizer, optimizer4d]

        self.metrics = metrics
        self.weight_initaliziton = weight_initaliziton
        self.generate_weights(weight_data_type)

        self.val_loss = None
        self.val_accuracy = None

    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        """Feed forward for the whole model

        Args:
            x (np.ndarray): x dataset

        Returns:
            np.ndarray: output of the model 
        """
        # We skip the first layer as it's the input layer
        for layer in self.layers[1:]:
            x = layer(x)
        return x

    def backpropagate(self, X: np.ndarray, y: np.ndarray, verbose: int = 2, epoch: int = 1, total_epochs: int = 100) -> None:
        """Backpropgate function to make the train function cleaner and better for future expansion

        Args:
            X (np.ndarray): X dataset
            y (np.ndarray): y dataset
            verbose (int, optional): What information should be printed out during training. 0 - None, 1 - Epoch/Epochs, 2 - Batch/Batches. Defaults to 2.
            epoch (int, optional): Current epoch we're on. Only used for printing out. Defaults to 1.
            total_epochs (int, optional): The amount of epochs we have to go through. Only used for printing out. Defaults to 100.
        """
        length_of_x = len(X)
        total_accuracy = 0
        losses = 0
        for i in range(length_of_x):
            start = time()
            yPred = self.feed_forward(X[i])

            # Accuracy calculation
            if self.metrics == "accuracy":
                if len(yPred) == 1:
                    total_accuracy += np.average(np.abs(y[i] - yPred) < 0.25)
                else:
                    total_accuracy += 1 if np.argmax(y[i]
                                                     ) == np.argmax(yPred) else 0

            gradient = self.loss_function.compute_derivative(y[i], yPred)
            # We skip over the input layer, as it doesn't have any parameters to update
            for layer in self.layers[-1:0:-1]:
                gradient = layer.backpropagate(gradient, self.optimizer)

            if verbose == 2:
                losses += self.loss_function.compute_loss(y[i], yPred)
                accuracy = total_accuracy / \
                    (i+1) if self.metrics == "accuracy" else None
                time_taken = time() - start
                # Note that we use (i + 1) as we want to divide the losses and accuracy by the amount of times they've been updated
                # self.print_progress(epoch+1, total_epochs, losses / (i+1),
                #                     accuracy, i+1, length_of_x, self.val_loss, self.val_accuracy, time_taken)
                self.print_progress(epoch+1, total_epochs, losses / (i+1),
                                    accuracy, i+1, length_of_x, time_taken)

    def _handle_callbacks(self, result: tuple[np.ndarray, np.ndarray] | None, callbacks: EarlyStopping | None) -> int:
        """Support function used for handling callbacks in train function. It either returns 0 - training continues, 1 - training stops

        Args:
            result (tuple[np.ndarray, np.ndarray] | None): Result from the callbacks.monitor() function. Either models weights or None
            callbacks (EarlyStopping | None): Callbacks we use during models training

        Returns:
            int: Information about training. 0 - continues, 1 - stops
        """
        if result is not None:
            if callbacks.restore_best_weights:
                for i, layer in enumerate(self.layers):
                    if result[0][i].size > 0:
                        layer.weights = result[0][i]
                        layer.biases = result[1][i]
            return 1
        return 0

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, callbacks: EarlyStopping = None, verbose: int = 1, validation_data: tuple[np.ndarray, np.ndarray] = None) -> np.ndarray | tuple:
        """Function to train the model. Remember to call the NN.compile() before calling this function as it won't work because we don't have weights. \n

        Args:
            X (np.ndarray): X dataset
            y (np.ndarray): y dataset
            epochs (int): number of iterations a model should do during training
            callbacks (EarlyStopping, optional): One of the callbacks implemented in callbacks.py although currently there's only early stopping in there. Defaults to None.
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
                self.print_progress(epoch+1, epochs, self.loss, self.accuracy,
                                    val_loss=self.val_loss, val_accuracy=self.val_accuracy)
        if validation_data is not None:
            return losses, val_losses
        return losses

    def evaluate(self, X: np.ndarray, y: np.ndarray, show_preds: bool = False, min_accuracy_error: float = 0.25) -> float:
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
        if self.metrics.find("accuracy") != -1:
            accuracy = 0
            if len(y.shape) == 1:
                if len(yPreds) == 1:
                    accuracy = np.average(
                        np.abs(y[i] - yPreds) < min_accuracy_error)
                else:
                    accuracy = 1 if np.argmax(y) == np.argmax(yPreds) else 0

            else:
                for i in range(len(y)):
                    if len(yPreds[i]) == 1:
                        accuracy += np.average(np.abs(y[i] - yPreds[i])
                                               < min_accuracy_error)
                    else:
                        accuracy += 1 if np.argmax(y[i]
                                                   ) == np.argmax(yPreds[i]) else 0

                accuracy /= len(y)

        return self.loss_function.compute_loss(y, yPreds), accuracy

    def save(self, file_path: str) -> None:
        """Function to save the models params into a file

        Args:
            file_path (str): File path where to save the model. Don't put the file extension as it is alredy handled by numpy.
            For example if you want to save the model at './saved_model' put that as the file_path, and numpy will add the extension
        """
        array_to_save = []
        for layer in self.layers:
            if any(isinstance(layer, layer_without_units) for layer_without_units in LAYERS_WITHOUT_UNITS):
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
