
import torch

nn = torch.nn
Tensor = torch.Tensor

class IModel(nn.Module):
    def __init__(self):
        super(IModel, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass