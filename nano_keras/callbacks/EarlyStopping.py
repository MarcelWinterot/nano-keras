import numpy as np
from nano_keras.callbacks import Callback
from nano_keras.layers import Layer, LayerWithParams


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
