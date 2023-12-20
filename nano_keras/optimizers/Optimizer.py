import numpy as np


class Optimizer:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _fill_array(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Support function to fill the m_w, v_w and m_b, v_b arrays if their shapes are smaller than the gradients

        Args:
            arr (np.ndarray): array to fill
            target_shape (tuple): what shape should the array have after filling it

        Returns:
            np.ndarray: filled array
        """
        arr_shape = arr.shape

        padding_needed = [max(0, target - current)
                          for target, current in zip(target_shape, arr_shape)]
        pad_width = [(0, padding) for padding in padding_needed]

        result = np.pad(arr, pad_width, mode='constant')
        return result

    def apply_gradients(self, weights_gradients: np.ndarray, bias_gradients: np.ndarray, weights: np.ndarray, biases: np.ndarray, update_biases: bool = True) -> tuple[np.ndarray, np.ndarray]:
        pass
