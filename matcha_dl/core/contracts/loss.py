from abc import abstractmethod

import torch
import torch.nn as nn

LOSS = "loss"

Tensor = torch.Tensor


class ILoss(nn.Module):
    """
    Abstract base class for torch loss functions.
    """

    def __init__(self, **kwargs):
        super(ILoss, self).__init__()

    @abstractmethod
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass of the loss function.

        Args:
            input (Tensor): The input tensor.
            target (Tensor): The target tensor.

        Returns:
            Tensor: The loss value as a tensor.
        """
        pass
