import torch.nn.functional as F
import torch

from matcha_dl.core.contracts.loss import ILoss, Tensor

from typing import Union, List


class BCELossWeighted(ILoss):
    """
    Binary Cross Entropy Loss Weighted.
    """

    def __init__(self, weight: List, device: Union[str, int], **kwargs) -> None:
        """
        Constructor for BCELossWeighted.

        Args:
            weight (List): A list of weights.
            device (int): The device on which to run the computations.
        """
        super(BCELossWeighted, self).__init__()

        self.dev = device
        self.weight = torch.tensor(weight).to(self.dev)
        self.reduction = "none"

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass of the loss function.

        Args:
            input (Tensor): The input tensor.
            target (Tensor): The target tensor.

        Returns:
            Tensor: The loss value as a tensor.
        """
        weight_ = self.weight[target.data.view(-1).to(self.dev).long()].view_as(target)

        loss = F.binary_cross_entropy(input, target, weight=None, reduction=self.reduction)
        loss_class_weighted = loss * weight_
        return loss_class_weighted.mean()
