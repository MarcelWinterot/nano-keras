from nano_keras.optimizers.Optimizer import Optimizer
from nano_keras.optimizers.Adadelta import Adadelta
from nano_keras.optimizers.Adagrad import Adagrad
from nano_keras.optimizers.Adam import Adam
from nano_keras.optimizers.NAdam import NAdam
from nano_keras.optimizers.RMSProp import RMSProp
from nano_keras.optimizers.SGD import SGD

OPTIMIZERS = {"adam": Adam(), "sgd": SGD(), "adagrad": Adagrad(
), "adadelta": Adadelta(), "rmsprop": RMSProp(), "nadam": NAdam()}
