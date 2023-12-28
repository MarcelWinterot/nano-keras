from nano_keras.optimizers.Optimizer import Optimizer
from nano_keras.optimizers.Adadelta import Adadelta
from nano_keras.optimizers.Adagrad import Adagrad
from nano_keras.optimizers.Adam import Adam
from nano_keras.optimizers.NAdam import NAdam
from nano_keras.optimizers.RMSProp import RMSProp
from nano_keras.optimizers.SGD import SGD
from nano_keras.optimizers.AdamW import AdamW
from nano_keras.optimizers.AdaMax import AdaMax
from nano_keras.optimizers.Adafactor import Adafactor
from nano_keras.optimizers.Ftrl import Ftrl

OPTIMIZERS = {"adam": Adam(), "sgd": SGD(), "adagrad": Adagrad(
), "adadelta": Adadelta(), "rmsprop": RMSProp(), "nadam": NAdam(), "adamw": AdamW(), "adamax": AdaMax(), "adafactor": Adafactor(), "Ftrl": Ftrl()}
