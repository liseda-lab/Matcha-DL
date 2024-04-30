from matcha_dl.core.contracts.stopper import IStopper


class EarlyStopping(IStopper):
    def __init__(self, tolerance: int = 5, min_delta: int = 0):
        """

        Args:
            tolerance (int, optional): The number of times the validation loss can be greater than the training loss before stopping. Defaults to 5.
            min_delta (int, optional): The minimum difference between validation loss and training loss to consider as improvement. Defaults to 0.
        """
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss: float, validation_loss: float) -> None:
        """
        Call an early stopper.

        Args:
            train_loss (float): The training loss.
            validation_loss (float): The validation loss.
        """
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
