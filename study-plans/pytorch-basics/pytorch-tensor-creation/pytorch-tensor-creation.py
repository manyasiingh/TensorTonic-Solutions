import torch

def create_tensor(method: str, shape: list, value: float = 0.0) -> torch.Tensor:
    if method == "zeros":
        return torch.zeros(shape).float()
    elif method == "ones":
        return torch.ones(shape).float()
    else:
        return torch.full((shape),value).float()
    
    pass
