import torch

class LoRA(torch.nn.Module):
    def __init__(self, rank, alpha, in_dim, out_dim):
        super().__init__()

        self.B = torch.nn.Parameter(torch.zeros(in_dim, rank))
        self.A = torch.nn.Parameter(torch.randn(rank, out_dim))
        self.scaling = alpha / rank

    def forward(self, x):
        print(f'X Size: {x.size()}')
        print(f'B Size: {self.B.size()}')
        print(f'A Size: {self.A.size()}')
        return  self.B @ self.A @ x * self.scaling  # not sure if x @ B @ A or B @ A @ x 

class LoRALinear(torch.nn.Module):

    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear 
        self.lora   = LoRA(
            rank=rank,
            alpha=alpha,
            in_dim=linear.in_features,
            out_dim=linear.out_features
            )
    
    def forward(self, x):
        return self.lora(x) + self.linear(x)


