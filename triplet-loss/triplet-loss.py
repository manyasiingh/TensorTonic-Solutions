import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    a_np = np.array(anchor)
    p_np = np.array(positive)
    n_np = np.array(negative)
    
    # Compute squared distances
    distap = np.sum(np.square(a_np - p_np), axis=-1)
    distan = np.sum(np.square(a_np - n_np), axis=-1)
    
    # Compute loss and take mean
    loss = np.maximum(0, distap - distan + margin)
    return float(np.mean(loss))