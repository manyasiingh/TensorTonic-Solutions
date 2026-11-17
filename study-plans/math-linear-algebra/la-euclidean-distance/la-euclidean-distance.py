import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    x = np.array(x)
    y = np.array(y)
    product = np.sqrt(np.sum(np.square(x-y)))
    return float(product)
    pass