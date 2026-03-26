import numpy as np

def cosine_similarity(a, b):
    num = np.dot(a , b)
    norm_a = np.sqrt(np.sum(np.square(a)))
    norm_b = np.sqrt(np.sum(np.square(b)))
    den = norm_a * norm_b
    if den == 0:
        return 0.0
    return num/den
    pass