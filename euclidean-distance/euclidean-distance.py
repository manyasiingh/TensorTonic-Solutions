import numpy as np

def euclidean_distance(x, y):
    x_np = np.array(x)
    y_np = np.array(y)
    diff = np.square(x_np - y_np)
    dist = np.sqrt(np.sum(diff))
    return dist
    pass