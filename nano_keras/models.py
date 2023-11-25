import numpy as np
from nano_keras.losses import LOSS_FUNCTIONS, Loss
from nano_keras.optimizers import OPTIMIZERS, Optimizer
from nano_keras.layers import Layer, LayerWithParams
from nano_keras.callbacks import Callback
from copy import deepcopy
from time import time


class NN:
    def __init__(self, layers: list[Layer] = [], name: str = "NN"):
        """NN init function. Simmilar to the keras.models.Sequential class. Simply add layers using NN.add(Layer)\n
        Note that the first layer is treated as the input layer so it's recommended to use layers.Input for it as you
        can modify the shape of the input data there

        Args:
            layers (list[Layer], optional): List of all the layers you want in the model. Note that you don't need to specify \n
            the layers here as you can use NN.add(Layer) to add another layer. Defaults to [].
            name (str, optional): Name of the model. Defaults to "NN".
        """
        self.name: str = name
        self.layers: list[Layer] = layers
        self.loss: float = 1e50
        self.accuracy: float = 0
        self.val_loss: float = None
        self.val_accuracy: float = None

    @staticmethod
    def __convert_size(size: int) -> str:
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
    def __convert_time(seconds: int) -> str:
        """Support function to convert ETA time into a more readable format

        Args:
            seconds (int): ETA time of the model

        Returns:
            str: Updated ETA time in a more readable format
        """
        hours, minutes = 0, 0

        if seconds >= 3600:
            hours = seconds // 3600
            seconds -= 3600 * hours

        if seconds > 60:
            minutes = seconds // 60
            seconds -= 60 * minutes

        if hours > 0:
            return f"{hours:.0f}:{minutes:.0f}:{seconds:.0f}"
        return f"{minutes:.0f}:{seconds:.0f}"

    def print_progress(self, epoch: int, total_epochs: int, loss: float, accuracy: float = None, batch: int = None, total_batches: int = None, time_taken: float = None) -> None:
        """Function that prints out current training progress

        Args:
            epoch (int): current epoch
            total_epochs (int): total number of epochs
            loss (float): loss calculated by NN.evaluate function
            accuracy (float, optional): accuracy of the model during training. Default to None.
            batch (int, optional): current batch. Will be set if verbose is set to 2 in the NN.train function. If it's None we ignore it during printing. Defaults to None.
            total_batches (int, optional): total number of batches. Will be set if verbose is set to 2 in the NN.train function. If it's None we ignore it during printing. Defaults to None.
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
            progress_info += f" - ETA: {self.__convert_time(time_taken * (total_batches - batch))}"

        if accuracy is not None:
            if type(accuracy) == np.nan:
                progress_info += f" - accuracy: {0:.3f}"
            else:
                progress_info += f" - accuracy: {accuracy:.3f}"

        if self.val_loss is not None:
            progress_info += f" - val_loss: {self.val_loss:.8f} - val_accuracy: {self.val_accuracy:.3f}"

        print(f"\r{progress_info}", end='')

    def get_weights(self) -> list[np.ndarray]:
        output = []
        empty_array = np.array([])
        for layer in self.layers:
            if isinstance(layer, LayerWithParams):
                output.append(np.copy(layer.weights))
            else:
                output.append(empty_array)

        return output

    def set_weights(self, weights: list[np.ndarray]) -> None:
        for i, layer in enumerate(self.layers):
            try:
                if isinstance(layer, LayerWithParams):
                    layer.weights = weights[i]
            except Exception as e:
                print(f"Exception occured when setting weights: {e}")
                exit(1)

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
            if isinstance(layer, LayerWithParams):
                totalParams += layer.weights.size + layer.biases.size
                paramsWeight += layer.weights.nbytes + layer.biases.nbytes
        print(f"{'='*line_length}")
        print(
            f"Total params: {totalParams} ({self.__convert_size(paramsWeight)})")
        print(f"{'_'*line_length}")

    def generate_weights(self, weight_data_type: np.float_) -> None:
        """Support function used in the compile function to generate model's weights.

        Args:
            weight_data_type (np.float_): numpy data type in which the models weights should be stored. Use only np.float_ data types.
        """
        for i in range(1, len(self.layers)):
            if isinstance(self.layers[i], LayerWithParams):
                try:
                    self.layers[i].generate_weights(
                        self.layers, i, weight_data_type)
                except Exception as e:
                    print(
                        f"Exception encountered when creating weights: {e}\nChange your achitecture and try again. If you think it's an error post an issue on github")
                    exit(1)

    def compile(self, loss_function: Loss | str = "mse", optimizer: Optimizer | str = "adam", metrics: str = "", weight_data_type: np.float_ = np.float64) -> None:
        """Function you should call before starting training the model, as we generate the weights in here, set the loss function and optimizer.

        Args:
            loss_function (Loss | str, optional): Loss function the model should use. You can pass either the name of it as a str or intialized class. Defaults to "mse".
            optimizer (Optimizer | str, optional): Optimizer the model should use when updating it's params. You can pass either the name of it as a str or initalized class. Defaults to "adam"
            metrics (str, optional): Paramter that specifies what metrics should the model use. Possible metrics are: accuracy. Defaults to "".
            weight_data_type (np.float_, optional): Data type you want the models weights to be. Use np.float_ types like np.float32 or np.float64. Defaults to np.float64.
        """
        self.loss_function: Loss = LOSS_FUNCTIONS[loss_function] if type(
            loss_function) == str else loss_function

        self.optimizer: Optimizer = OPTIMIZERS[optimizer] if type(
            optimizer) == str else optimizer
        optimizer4d = deepcopy(self.optimizer)
        self.optimizer: list[Optimizer] = [self.optimizer, optimizer4d]

        self.metrics: str = metrics
        self.generate_weights(weight_data_type)

    def feed_forward(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        """Feed forward for the whole model

        Args:
            x (np.ndarray): x dataset
            isTraining (bool): Changes the behavior of a few layers like Dropout so that it gives better results

        Returns:
            np.ndarray: output of the model
        """
        for layer in self.layers:
            x = layer(x, is_training)
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
            yPred = self.feed_forward(X[i], True)

            if self.metrics == "accuracy":
                total_accuracy += self.__calculate_accuracy(yPred, y[i])

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
                self.print_progress(epoch, total_epochs, losses / (i+1),
                                    accuracy, i+1, length_of_x, time_taken)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, callbacks: Callback = None, verbose: int = 1, validation_split: float = 0, validation_data: tuple[np.ndarray, np.ndarray] = None) -> np.ndarray | tuple:
        """Function to train the model. Remember to call the NN.compile() before calling this function as it won't work because we don't have weights. \n

        Args:
            X (np.ndarray): X dataset
            y (np.ndarray): y dataset
            epochs (int): number of iterations a model should do during training
            callbacks (Callback, optional): One of the callbacks implemented in callbacks.py although currently there's only early stopping in there. Defaults to None.
            verbose (int, optional): Parameter to control what the model prints out during training. 0 - nothing, 1 - only epoch/epochs, 2 - all the useful information. Defaults to 1.
            validation_split (float, optional): How much of the training data do you want to split for validation. Only works if validation_data is not assigned any value. Remember that it must be between 0 and 1. Defaults to 0
            validation_data (tuple, optional): Validation data a model should use to check the validation loss and accuracy. It should be a tuple of X and y. Default to None.

        Returns:
            np.ndarray | tuple: Either the losses model's had during training or both the losses and val_losses if validation_data is set.
        """
        losses = np.ndarray((epochs))
        val_losses = np.ndarray((epochs))

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        if validation_data is None and 0 < validation_split < 1:
            split_index = int(len(X) * (1 - validation_split))
            X, X_val = X[:split_index], X[split_index:]
            y, y_val = y[:split_index], y[split_index:]
            validation_data = (X_val, y_val)

        training_active: bool = [True]

        for epoch in range(1, epochs+1):
            callbacks.on_epoch_start(
                epoch=epoch, lr=self.optimizer[0].learning_rate, optimizers=self.optimizer)

            self.backpropagate(X, y, verbose, epoch, epochs)
            self.loss, self.accuracy = self.evaluate(X, y)

            if validation_data is not None:
                self.val_loss, self.val_accuracy = self.evaluate(
                    validation_data[0], validation_data[1])

                self._metrics = {"loss": self.loss, "accuracy": self.accuracy,
                                 "val_loss": self.val_loss, "val_accuracy": self.val_accuracy}

                val_losses[epoch] = self.val_loss
            else:
                self._metrics = {"loss": self.loss, "accuracy": self.accuracy}

            callbacks.on_epoch_end(
                metrics=self._metrics, layers=self.layers, training_active=training_active)

            if not training_active[0]:
                break

            losses[epoch] = self.loss
            if verbose == 1:
                self.print_progress(epoch+1, epochs, self.loss, self.accuracy,
                                    val_loss=self.val_loss, val_accuracy=self.val_accuracy)
        if validation_data is not None:
            return losses, val_losses
        return losses

    def __calculate_accuracy(self, yPreds: np.ndarray, yTrue: np.ndarray, min_accuracy_error: float = 0.25) -> float:
        """Function to generate the accuracy given yPreds and yTrue dataset. It's earned a function for itself as I want the code to be cleaner
        as we have repeating code in NN.backpropagate() and NN.evaluate()

        Args:
            yPreds (np.ndarray): Predictions of the model
            yTrue (np.ndarray): Correct values the model should have predicted
            min_accuracy_error (float): The minimum admissible error threshold for considering an answer as correct.

        Returns:
            float: Calculated accuracy
        """
        accuracy = 0
        if len(yTrue.shape) == 1:
            if len(yPreds) == 1:
                return np.average(
                    np.abs(yTrue - yPreds) < min_accuracy_error)

            return 1 if np.argmax(yTrue) == np.argmax(yPreds) else 0

        for i in range(len(yTrue)):
            if len(yPreds[i]) == 1:
                accuracy += np.average(np.abs(yTrue[i] - yPreds[i])
                                       < min_accuracy_error)
            else:
                accuracy += 1 if np.argmax(yTrue[i]
                                           ) == np.argmax(yPreds[i]) else 0

        accuracy /= len(yTrue)

        return accuracy

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
            yPreds[i] = self.feed_forward(x, False)

        if show_preds:
            print(yPreds)

        accuracy = None
        if self.metrics.find("accuracy") != -1:
            accuracy = self.__calculate_accuracy(yPreds, y, min_accuracy_error)

        return self.loss_function.compute_loss(y, yPreds), accuracy

    def save(self, file_path: str) -> None:
        """Function to save the models params into a file

        Args:
            file_path (str): File path where to save the model. Don't put the file extension as it is alredy handled by numpy.
            For example if you want to save the model at './saved_model' put that as the file_path, and numpy will add the extension
        """
        array_to_save = []
        for layer in self.layers:
            if not isinstance(layer, LayerWithParams):
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
