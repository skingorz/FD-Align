import torch
import torch.nn as nn
import torch.nn.functional as F
from architectures.utils import L2SquareDist
from torch import Tensor

class CoOp_head(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

def create_model(metric: str = "cosine", 
        scale_cls: int =10.0, 
        learn_scale: bool = True, 
        normalize: bool = True):
    return CoOp_head
