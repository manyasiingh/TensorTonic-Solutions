import torch

def tensor_op(x: torch.Tensor, y: torch.Tensor, op: str) -> torch.Tensor:
    """
    Returns the operation result as a float32 tensor.
    """
    if op == 'add':
        return torch.add(x,y).float()
    elif op == "multiply":
        return torch.mul(x,y).float()
    elif op == "matmul":
        return torch.matmul(x,y).float()
    elif op == "power":
        return torch.pow(x,y).float()
    else:
        return torch.max(x,y).float()
    pass
