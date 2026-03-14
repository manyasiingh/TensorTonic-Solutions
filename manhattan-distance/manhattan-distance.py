import numpy as np

def manhattan_distance(x, y):
    x_np = np.array(x)
    y_np = np.array(y)
    dist = np.abs(x_np - y_np)
    man_dist = np.sum(dist)
    return float(man_dist)
    pass