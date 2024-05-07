import itertools
from typing import List, Optional

from matcha_dl.core.contracts.model import IModel, Tensor, nn


def pairwise(iterable: List[int]) -> zip:
    """
    Returns a zip object containing pairs of consecutive elements in the iterable.

    Args:
        iterable (List[int]): A list of integers.

    Returns:
        zip: A zip object containing pairs of consecutive elements in the iterable.
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class MlpClassifier(IModel):

    def __init__(
        self, layers: List[int], n: Optional[int] = 5, n_classes: Optional[int] = 1, **kwargs
    ):
        """
        Parameters:
            layers (List[int]): The sizes of the hidden layers.
            n (int): The size of the input layer.
            n_classes (int): The size of the output layer.
        """
        super(MlpClassifier, self).__init__()

        layers.insert(0, n)

        _layers = []

        for in_feats, out_feats in pairwise(layers):
            _layers.append(nn.Linear(in_feats, out_feats))

            _layers.append(nn.ReLU())

        self._hidden_layers = nn.Sequential(*_layers)

        self.classify = nn.Linear(layers[-1], n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters:
            x (Tensor): The input to the MLP.

        Returns:
            Tensor: The output of the MLP.
        """

        return self.sigmoid(self.classify(self._hidden_layers(x)))
