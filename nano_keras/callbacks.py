import numpy as np
from nano_keras.layers import Layer


class EarlyStopping:
    def __init__(self, patience: int, value_to_monitor: str = "loss", min_delta: float = 0.0001, restore_best_weights: bool = False) -> None:
        """Early stopping implementation using just python and numpy

        Args:
            patience (int): For how many epochs can the moniotred value degrade before stopping training
            value_to_monitor (int, optional): what should the model watch during training. Possible value_to_monitor options are: metric, accuracy, val_metric, val_accuracy. Default to loss.
            min_delta (float, optional): By how much must the value_to_monitored improve in order for it to be classiffied as an improvement. Default to 0.0001
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

    def update_weights(self, layers: list) -> None:
        """Support function to get the weights of the model

        Args:
            layers (list): List of all the layers in a model
        """
        self.weights = []
        self.biases = []
        for layer in layers:
            self.weights.append(layer.weights)
            self.biases.append(layer.biases)

    def set_weights(self, layers: list[Layer]) -> None:
        for i, layer in enumerate(layers):
            layer.weights = self.weights[i]
            layer.biases = self.biases[i]

    def watch(self, metric: np.ndarray, layers: list[Layer]) -> bool:
        """Function to watch over the models metric to measure and see if it's getting better or worse

        Args:
            metric (np.ndarray): New metric
            layers (list): List of all layers in a model

        Returns:
            bool: Retuns True if the training has finished and False if the training continoues
        """
        if metric is None:
            if self.value_to_monitor.find("accuracy") != -1:
                metric = 0
            elif self.value_to_monitor.find("loss") != -1:
                metric = 1e50
            else:
                raise "self.value_to_monitor must be either accuracy, val_accuracy, loss or val_loss"

        if (abs(metric - self.metric) < self.min_delta and self.value_to_monitor.find("loss") != -1) or (metric > self.metric and self.value_to_monitor.find("accuracy") != -1):
            self.metric = metric
            self.update_weights(layers)
            return False

        self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best_weights:
                self.set_weights(layers)
            return True
