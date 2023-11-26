import numpy as np
from nano_keras.layers import Layer, LayerWithParams
import csv


class Callback:
    def __init__(self) -> None:
        pass

    def on_epoch_start(self, *args, **kwargs) -> None:
        return

    def on_epoch_end(self, *args, **kwargs) -> None:
        return

    def on_batch_start(self, *args, **kwargs) -> None:
        return

    def on_batch_end(self, *args, **kwargs) -> None:
        return


class EarlyStopping(Callback):
    def __init__(self, patience: int, value_to_monitor: str = "loss", min_delta: float = 0.0001, restore_best_weights: bool = False) -> None:
        """Initalizer for the early stopping callback. It's used to stop the training if the model
        doesn't improve or it's improvement is too small

        Args:
            patience (int): For how many epochs can the moniotred value degrade before stopping training\n
            value_to_monitor (int, optional): what should the model watch during training.
            Possible value_to_monitor options are: metric, accuracy, val_metric, val_accuracy. Default to loss.\n
            min_delta (float, optional): By how much must the value_to_monitored improve
            in order for it to be classiffied as an improvement. Default to 0.0001\n
            restore_best_weights (bool, optional): Should the best weights be restored when the training is finished. Defaults to False.
        """
        self.patience: int = patience
        self.value_to_monitor: str = value_to_monitor
        self.min_delta: float = min_delta
        self.restore_best_weights: bool = restore_best_weights
        self.metric: float = float(
            1e50) if value_to_monitor.find("loss") != -1 else 0
        self.weights: list = []
        self.biases: list = []
        self.counter: int = 0

    def __update_weights(self, layers: list[Layer]) -> None:
        """Support function to get the weights of the model

        Args:
            layers (list[Layer]): List of all the layers in a model
        """
        self.weights = []
        self.biases = []
        empty_array = np.array([])
        for layer in layers:
            if isinstance(layer, LayerWithParams):
                self.weights.append(layer.weights)
                self.biases.append(layer.biases)
                continue
            self.weights.append(empty_array)
            self.biases.append(empty_array)

    def __set_weights(self, layers: list[Layer]) -> None:
        """Support function to set the weights of the model after training has finished

        Args:
            layers (list[Layer]): List of all layers in a model
        """
        for i, layer in enumerate(layers):
            layer.weights = self.weights[i]
            layer.biases = self.biases[i]

    def on_epoch_end(self, *args, **kwargs) -> None:
        """Function used to monitor the training and decide wheter it should stop or no\n
        You need to pass those kwargs in order for this to work: metrics, layers
        """
        metric = kwargs['metrics'][self.value_to_monitor]

        if (abs(metric - self.metric) < self.min_delta and self.value_to_monitor.find("loss") != -1) or (metric > self.metric and self.value_to_monitor.find("accuracy") != -1):
            self.metric = metric
            self.__update_weights(kwargs['layers'])
            return

        self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best_weights:
                self.__set_weights(kwargs['layers'])
            kwargs['training_active'][0] = False


class LearningRateScheduler(Callback):
    def __init__(self, schedule: callable) -> None:
        """Initalizer for the LearningRateScheduler callback. It's used to change the\n
        learning rate of the model depending on the given function, called schedule.

        Args:
            schedule (callable): Functon used to change models learning rate during the training\n
            The function needs to take epoch(int) and learning rate(float) as it's parameters, and return\n
            the new learning rate(float)
        """
        self.schedule: callable = schedule

    def on_epoch_start(self, *args, **kwargs) -> None:
        """Function used to update the models learning rate at the start of each epoch\n
        You need to pass epoch, lr and optimizers in the kwargs in order for this to work
        """
        lr = self.schedule(kwargs['epoch'], kwargs['lr'])

        if 'optimizers' in kwargs:
            kwargs['optimizers'][0].learning_rate = lr
            kwargs['optimizers'][1].learning_rate = lr


class CSVLogger(Callback):
    def __init__(self, filename: str, append: bool = False) -> None:
        """Initalizr for the CSVLogger callback. It's used to log information about training\n
        into a .csv file, so you can check it later. It logs the information after the batch is finished\n
        Note that if you don't set metrics="accuracy" in the model.compile(), accuracy columns will be empty.

        Args:
            filename (str): Filename where you want to set the logs. You can but don't need to add the .csv extension
            append (bool, optional): If set to true CSVLogger will add to the already created file. Defaults to False.
        """
        self.filename = filename if '.csv' in filename else f"{filename}.csv"

        if not append:
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    ['Epoch', 'Batch', 'Accuracy', 'Loss', 'Time taken'])

    def on_batch_end(self, *args, **kwargs) -> None:
        """Function to save the information about training in the .csv file.\n
        It saves all the parameters given to the function in the *args.
        """
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(args)
