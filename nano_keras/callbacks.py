import numpy as np
from nano_keras.layers import Layer, LayerWithParams


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
        empty_array = np.array([])
        for layer in layers:
            if isinstance(layer, LayerWithParams):
                self.weights.append(layer.weights)
                self.biases.append(layer.biases)
                continue
            self.weights.append(empty_array)
            self.biases.append(empty_array)

    def set_weights(self, layers: list[Layer]) -> None:
        for i, layer in enumerate(layers):
            layer.weights = self.weights[i]
            layer.biases = self.biases[i]

    def on_epoch_end(self, *args, **kwargs) -> None:
        metric = kwargs['metrics'][self.value_to_monitor]

        if (abs(metric - self.metric) < self.min_delta and self.value_to_monitor.find("loss") != -1) or (metric > self.metric and self.value_to_monitor.find("accuracy") != -1):
            self.metric = metric
            self.update_weights(kwargs['layers'])
            return

        self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best_weights:
                self.set_weights(kwargs['layers'])
            kwargs['training_active'][0] = False


class LearningRateScheduler(Callback):
    def __init__(self, schedule: callable) -> None:
        self.schedule: callable = schedule

    def on_epoch_start(self, *args, **kwargs) -> None:
        lr = self.schedule(kwargs['epoch'], kwargs['lr'])

        if 'optimizers' in kwargs:
            kwargs['optimizers'][0].learning_rate = lr
            kwargs['optimizers'][1].learning_rate = lr
