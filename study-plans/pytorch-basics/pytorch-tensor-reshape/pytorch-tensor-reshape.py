import torch

def reshape_tensor(x, op):
    x = torch.tensor(x, dtype = torch.float32)
    if op == "flatten":
        return x.flatten().tolist()
    elif op == "squeeze":
        return x.squeeze().tolist()
    else:
        return x.T.tolist()
    pass
