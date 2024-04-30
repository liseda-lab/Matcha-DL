from abc import abstractmethod

STOPPER = "stopper"


class IStopper:
    @abstractmethod
    def __init__(self, tolerance: int, min_delta: int):
        """

        Args:
            tolerance (int): The number of times the validation loss can be greater than the training loss before stopping.
            min_delta (int): The minimum difference between validation loss and training loss to consider as improvement.
        """
        pass

    @abstractmethod
    def __call__(self, train_loss: float, validation_loss: float) -> None:
        """
        Abstract method for calling a stopper.

        Args:
            train_loss (float): The training loss.
            validation_loss (float): The validation loss.
        """
        pass
