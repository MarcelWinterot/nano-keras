from nano_keras.callbacks import Callback


class LearningRateScheduler(Callback):
    """LearningRateScheduler callback used to change the learning rate of the model during training\n
    New learning rate of the model is calculated by the given function, called schedule.
    """

    def __init__(self, schedule: callable) -> None:
        """Initalizer for the LearningRateScheduler callback. 

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
