import numpy as np

def log_rotation(R):
    theta = np.arccos(max(-1., min(1., (np.trace(R) - 1.) / 2.)))
    if theta < 1e-12:
        return np.zeros((3, 1))
    mat = R - R.T
    r = np.array([mat[2, 1], mat[0, 2], mat[1, 0]]).reshape((3, 1))
    return theta / (2. * np.sin(theta)) * r