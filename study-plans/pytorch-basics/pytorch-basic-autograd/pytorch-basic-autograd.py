import torch

def compute_gradient(values: torch.Tensor) -> torch.Tensor:
    values.requires_grad_(True)
    y = (values**(3) + 2 * values)
    y.sum().backward()
    return values.grad
    pass
