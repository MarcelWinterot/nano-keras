from nano_keras.losses.Loss import Loss
from nano_keras.losses.MAE import MAE
from nano_keras.losses.MSE import MSE
from nano_keras.losses.BCE import BCE
from nano_keras.losses.CCE import CCE
from nano_keras.losses.Hinge import Hinge
from nano_keras.losses.Huber import Huber


LOSS_FUNCTIONS = {"mae": MAE(), "mse": MSE(), "bce": BCE(), "cce": CCE(),
                  "huber": Huber(), "hinge": Hinge()}
