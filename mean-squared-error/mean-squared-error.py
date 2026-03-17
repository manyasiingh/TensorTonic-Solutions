import numpy as np

def mean_squared_error(y_pred, y_true):
    y_pred_np = np.array(y_pred)
    y_true_np = np.array(y_true)
    mse = (np.sum(np.square(y_pred_np - y_true_np)))
    return float(mse/y_pred_np.size)
    pass
