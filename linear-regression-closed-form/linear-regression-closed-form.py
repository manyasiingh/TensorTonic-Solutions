import numpy as np
#fully vectorised use of numpy no forr loops
def linear_regression_closed_form(X, y):
    X_np = np.array(X)
    y_np = np.array(y)
    X_trans = np.transpose(X_np)
    X_pro = np.dot(X_trans , X_np)
    Xy = np.dot(X_trans , y)
    X_inv = np.linalg.inv(X_pro)
    w = np.dot(Xy , X_inv)
    return w
    pass