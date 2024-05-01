import torch.nn.functional as F

from matcha_dl.core.contracts.loss import ILoss, Tensor


class BCELossWeighted(ILoss):
    """
    Binary Cross Entropy Loss Weighted.
    """

    def __init__(self, weight: Tensor, device: Tensor, **kwargs) -> None:
        """
        Constructor for BCELossWeighted.

        Args:
            weight (torch.Tensor): The weight tensor.
            device (torch.device): The device on which to run the computations.
        """
        super(BCELossWeighted, self).__init__()

        self.dev = device
        self.weight = weight.to(self.dev)
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
