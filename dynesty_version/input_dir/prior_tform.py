import numpy as np


#########################
### 2D Gaussian Ring
#########################


def twod_gaussring_ptform(u):
    """
    Prior transform function for dynesty nested sampling of the 2D Gaussian Ring model.

    Transforms uniform unit hypercube parameters u in [0, 1] to physical prior ranges:
        model_fits_ranges = [
            [-5, 15],   # peak
            [0, 10],    # sigma
            [0, 20],    # ring_rad
            [0, 90],    # inclination
            [0, 180],   # posangle
            [-5, 5],    # dRA
            [-5, 5]     # dDec
        ]

    Args:
        u (array-like): Parameters in unit hypercube space [0, 1].

    Returns:
        v (np.ndarray): Transformed parameters in physical parameter space.
    """
    u = np.asarray(u)
    prior_ranges = np.array([
        [-5, 15],
        [0, 10],
        [0, 20],
        [0, 90],
        [0, 180],
        [-5, 5],
        [-5, 5]
    ], dtype=float)

    low = prior_ranges[:, 0]
    high = prior_ranges[:, 1]

    v = low + u * (high - low)
    
    return v


#########################
### 2D Gaussian Ring + 1 Gaussian Blob
#########################


def twod_gauss1blob_ptform(u):
    """
    Prior transform function for dynesty nested sampling of the 2D Gaussian Ring model.

    Transforms uniform unit hypercube parameters u in [0, 1] to physical prior ranges:
        model_fits_ranges = [
            [-5, 15],   # peak
            [0, 10],    # sigma
            [0, 20],    # ring_rad
            [0, 90],    # inclination
            [0, 180],   # posangle
            [-5, 5],    # dRA
            [-5, 5]     # dDec
        ]

    Args:
        u (array-like): Parameters in unit hypercube space [0, 1].

    Returns:
        v (np.ndarray): Transformed parameters in physical parameter space.
    """
    u = np.asarray(u)
    prior_ranges = np.array([
        [-5, 15],
        [0, 10],
        [0, 20],
        [0, 90],
        [0, 180],
        [-5, 5],
        [-5, 5],
        [-5, 15],
        [0, 0.6],
        [5, 8],
        [150, 200]
    ], dtype=float)

    low = prior_ranges[:, 0]
    high = prior_ranges[:, 1]

    v = low + u * (high - low)
    
    return v
