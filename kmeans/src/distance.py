import numpy as np

# squared euclidean distance
def squared_euclidean_distance(x, y):
    return np.sum((x - y) ** 2, axis=-1)

# minkowski distance
def minkowski_distance(x, y, p):
    return np.sum(np.abs(x - y) ** p, axis=-1) ** (1 / p)

# cosine distance
def cosine_distance(x, y):
    dot_product = np.sum(x * y, axis=-1)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    
    # menambahkan epsilon kecil untuk mencegah division by zero
    epsilon = 1e-10
    return 1 - (dot_product / (norm_x * norm_y + epsilon))

# chebyshev distance.
def chebyshev_distance(x, y):
    return np.max(np.abs(x - y), axis=-1)

# manhattan distance
def manhattan_distance(x, y):
    return np.sum(np.abs(x - y), axis=-1)