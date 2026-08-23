import torch

def create_tensor(method, shape, value=0.0):
    if method == "zeroes":
        return torch.zeroes(shape).tolist()
    elif method == "ones":
        return torch.ones(shape).tolist()
    else:
        return torch.full(shape, value).tolist()
    
    pass