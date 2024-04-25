import torch

class LoRALayer(torch.nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer which applies linear transform to input,
    where the transform matrix Delta(W) is decomposed Delta(W) = AB. Here A, B
    are smaller matrices, i.e. we assume Delta(W) is low rank. 

    Args:
        rank (int)   :    The rank to decompose the adaptation matrix into, effectively controlling the number of parameters.
        alpha (float): Scaling factor for the adaptation output, used to control the magnitude of the transformation.
        in_dim (int) :  Dimension of the input vector.
        out_dim (int): Dimension of the output vector.

    Attributes:
        A (torch.nn.Parameter): low-rank matrix of dimension (in_dim, rank).
        B (torch.nn.Parameter): low-rank matrix of dimension (rank, out_dim).
        scaling (float)       : scaling factor applied to the transformation output.
    """
    def __init__(self, rank, alpha, in_dim, out_dim):
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank 

    def forward(self, x):
        """
        Forward pass of the LoRA layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_dim).
        """
        x = self.scaling * (x @ self.A @ self.B)
        return x


class LinearLoRALayer(torch.nn.Module):
    """
    Extends a standard linear layer with a LoRA layer.

    Args:
        linear (torch.nn.Linear): Original linear layer to be extended.
        rank (int)              : The rank for the low-rank matrices in the LoRA layer.
        alpha (float)           : Scaling factor for the LoRA output.

    Attributes:
        linear (torch.nn.Linear): Standard linear layer.
        lora (LoRALayer)        : Low-rank adaptation layer.
    """
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            rank=rank,
            alpha=alpha,
            in_dim=linear.in_features,
            out_dim=linear.out_features)

    def forward(self, x):
        """
        Forward pass of the LinearLoRALayer, which combines outputs from the original linear layer and the LoRA layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features) where `in_features` should match `self.linear.in_features`.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features) where `out_features` is `self.linear.out_features`.

        Uses distributive law: (W + Delta(W))x = (W + AB)x = Wx + ABx, which enables quick swapping or LoRA layers
        """
        return self.linear(x) + self.lora(x)
