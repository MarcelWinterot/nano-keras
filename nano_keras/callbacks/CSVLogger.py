from nano_keras.callbacks import Callback
import csv


class CSVLogger(Callback):
    def __init__(self, filename: str, append: bool = False) -> None:
        """Initalizr for the CSVLogger callback. It's used to log information about training\n
        into a .csv file, so you can check it later. It logs the information after the batch is finished\n
        Note that if you don't set metrics="accuracy" in the model.compile(), accuracy columns will be empty.

        Args:
            filename (str): Filename where you want to set the logs. You can but don't need to add the .csv extension
            append (bool, optional): If set to true CSVLogger will add to the already created file. Defaults to False.
        """
        self.filename = filename if '.csv' in filename else f"{filename}.csv"

        if not append:
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    ['Epoch', 'Batch', 'Accuracy', 'Loss', 'Time taken'])

    def on_batch_end(self, *args, **kwargs) -> None:
        """Function to save the information about training in the .csv file.\n
        It saves all the parameters given to the function in the *args.
        """
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(args)
