class Callback:
    """Base class used to build new callbacks.
    """

    def __init__(self) -> None:
        """Initailizer for Callback class
        """
        pass

    def on_epoch_start(self, *args, **kwargs) -> None:
        """Function called at the start of each epoch
        """
        return

    def on_epoch_end(self, *args, **kwargs) -> None:
        """Function called at the end of each epoch
        """
        return

    def on_batch_start(self, *args, **kwargs) -> None:
        """Function called at the start of each batch
        """
        return

    def on_batch_end(self, *args, **kwargs) -> None:
        """Function called at the end of each batch
        """
        return
