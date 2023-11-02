import numpy as np


class EarlyStopping:
    def __init__(self, patience: int, monitor: str = "metric", min_delta: float = 0.0001, restore_best_weights: bool = False) -> None:
        """Early stopping implementation using just python and numpy

        Args:
            patience (int): For how many epochs can the moniotred value degrade before stopping training
            monitor (int, optional): what should the model watch during training. Possible monitors are: metric, accuracy, val_metric, val_accuracy. Default to metric.
            min_delta (float, optional): By how much must the monitored value improve in order for it to be classiffied as an improvement. Default to 0.0001
            restore_best_weights (bool, optional): Should the best weights be restored when the training is finished. Defaults to False.
        """
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.metric = float(1e50)
        self.weights = []
        self.biases = []
        self.counter = 0

    def get_models_weights(self, layers: list) -> None:
        for layer in layers:
            try:
                self.weights.append(layer.weights)
                self.biases.append(layer.biases)
            except:
                self.weights.append([])
                self.biases.append([])

    def watch(self, metric: np.ndarray, layers: list) -> tuple | None:
        """
        Plans for the algorithm:
        We will check if the new metric has improved to the currently stored one
        If it has improved it will be stored in the memory with the current weights, and counter will be reset to 0
        If it has not improved we will update the counter by 1 and check if the counter == patience, if so end the training and return the best weights

        Args:
            metric (np.ndarray): New metric
            weights (np.ndarray): Our models weights
            biases (np.ndarray): Our models biases

        Returns:
            tuple | None: Either the weights and biases in a tuple if the training has finished or nothing if the training continues
        """
        if metric is None:
            if self.monitor.find("accuracy") != -1:
                metric = 0
            elif self.monitor.find("loss") != -1:
                metric = 1e50
            else:
                raise "self.monitor must be either accuracy, val_accuracy, loss or val_loss"

        if (metric - self.metric < self.min_delta and (self.monitor == "loss" or self.monitor == "val_loss")) or (self.metric > metric and (self.monitor == "accuracy" or self.monitor == "val_accuracy")):
            self.metric = metric
            self.get_models_weights(layers)
            return None

        self.counter += 1
        if self.counter >= self.patience:
            return (self.weights, self.biases) if self.restore_best_weights else ()
