"""
Prior Transformations for Dynesty Nested Sampling
===================================================
In Bayesian nested sampling (Dynesty), sampling is initially performed on an
$N$-dimensional unit hypercube where each coordinate $u_i \in [0, 1]$ represents
a cumulative probability value.

A **prior transform function** $\theta = F^{-1}(u)$ maps these normalized unit cube
coordinates to physical parameter values according to their prior probability
distributions. For uniform priors $U(a, b)$, the transformation is:
    $\theta = a + u \cdot (b - a)$

This module defines prior transform functions for various geometric models of
galaxy continuum emission (Gaussian rings and embedded compact clumps/blobs).
"""

import numpy as np


#########################
### 2D Gaussian Ring
#########################

def twod_gaussring_ptform(u):
    """
    Prior transform for the 2D Gaussian Ring model (7 parameters).

    Transforms unit hypercube coordinates $u \in [0, 1]^7$ into physical
    parameters assuming uniform priors across specified intervals.

    Parameters
    ----------
    u : array_like, shape (7,)
        Normalized coordinates on the unit hypercube, where each $u_i \in [0, 1]$.
        Order of parameters:
            0: Peak brightness log10(Jy/sr)
            1: Ring Gaussian radial width (arcsec)
            2: Ring center radius (arcsec)
            3: Disk inclination angle (degrees, 0 = face-on, 90 = edge-on)
            4: Disk position angle (degrees, East of North)
            5: Right Ascension offset (arcsec)
            6: Declination offset (arcsec)

    Returns
    -------
    v : numpy.ndarray, shape (7,)
        Physical model parameters corresponding to the unit coordinates $u$.
        Prior ranges:
            - Peak:      log(Jy/sr)
            - Width:     arcsec
            - Ring Rad:  arcsec
            - Inc:       degrees
            - PA:        degrees
            - Offset RA: arcsec
            - Offset Dec:arcsec
    """
    u = np.asarray(u)
    prior_ranges = np.array([
        [-5, 15],   # Peak [log(Jy/sr)]
        [0, 10],    # Width [arcsec]
        [0, 20],    # Ring Rad [arcsec]
        [0, 90],    # Inclination [deg]
        [0, 180],   # Position Angle [deg]
        [-5, 5],    # Offset RA [arcsec]
        [-5, 5]     # Offset Dec [arcsec]
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
    Prior transform for the 2D Gaussian Ring + 1 Gaussian Blob model (11 parameters).

    Transforms unit hypercube coordinates $u \in [0, 1]^{11}$ into physical parameters
    for a ring profile superimposed with a single compact Gaussian brightness knot.

    Parameters
    ----------
    u : array_like, shape (11,)
        Normalized coordinates on the unit hypercube ($u_i \in [0, 1]$).
        Order of parameters:
            0-6:  Ring parameters (Peak, Width, Radius, Inc, PA, dRA, dDec)
            7:    Blob 1 peak brightness log10(Jy/sr)
            8:    Blob 1 Gaussian width (arcsec)
            9:    Blob 1 radial distance from ring center (arcsec)
            10:   Blob 1 azimuthal position angle (degrees)

    Returns
    -------
    v : numpy.ndarray, shape (11,)
        Physical model parameters. Prior ranges:
            - Ring Peak:       log(Jy/sr)
            - Ring Width:      arcsec
            - Ring Radius:     arcsec
            - Inclination:     degrees
            - Position Angle:  degrees
            - Offset RA:       arcsec
            - Offset Dec:      arcsec
            - Blob 1 Peak:     log(Jy/sr)
            - Blob 1 Width:    arcsec
            - Blob 1 Distance: arcsec
            - Blob 1 Angle:    degrees
    """
    u = np.asarray(u)
    prior_ranges = np.array([
        [-5, 15],   # Ring Peak [log(Jy/sr)]
        [0, 10],    # Ring Width [arcsec]
        [0, 20],    # Ring Rad [arcsec]
        [0, 90],    # Inclination [deg]
        [-10, 10],   # Position Angle [deg]
        [-5, 5],    # Offset RA [arcsec]
        [-5, 5],    # Offset Dec [arcsec]
        [-5, 15],   # Blob 1 Peak [log(Jy/sr)]
        [0, 0.6],   # Blob 1 Width [arcsec]
        [5, 8],     # Blob 1 Dist [arcsec]
        [150, 200]  # Blob 1 Angle [deg]
    ], dtype=float)

    low = prior_ranges[:, 0]
    high = prior_ranges[:, 1]

    v = low + u * (high - low)
    
    return v


#########################
### 2D Gaussian Ring + 3 Gaussian Blobs
#########################

def twod_gauss3blob_ptform(u):
    """
    Prior transform for the 2D Gaussian Ring + 3 Gaussian Blobs model (19 parameters).

    Transforms unit hypercube coordinates $u \in [0, 1]^{19}$ into physical parameters
    for a ring profile superimposed with three distinct compact emission clumps.

    Parameters
    ----------
    u : array_like, shape (19,)
        Normalized coordinates on the unit hypercube ($u_i \in [0, 1]$).
        Order of parameters:
            0-6:   Ring parameters (Peak, Width, Radius, Inc, PA, dRA, dDec)
            7-10:  Blob 1 parameters (Peak, Width, Distance, Angle)
            11-14: Blob 2 parameters (Peak, Width, Distance, Angle)
            15-18: Blob 3 parameters (Peak, Width, Distance, Angle)

    Returns
    -------
    v : numpy.ndarray, shape (19,)
        Physical model parameters. Prior ranges:
            - Ring Peak:       log(Jy/sr)
            - Ring Width:      arcsec
            - Ring Radius:     arcsec
            - Inclination:     degrees
            - Position Angle:  degrees
            - Offset RA:       arcsec
            - Offset Dec:      arcsec
            - Blob 1-3 Peak:   log(Jy/sr)
            - Blob 1-3 Width:  arcsec
            - Blob 1-3 Dist:   arcsec
            - Blob 1-3 Angle:  degrees
    """
    u = np.asarray(u)
    prior_ranges = np.array([
        [-5, 15],   # Ring Peak [log(Jy/sr)]
        [0, 10],    # Ring Width [arcsec]
        [0, 20],    # Ring Rad [arcsec]
        [0, 90],    # Inclination [deg]
        [-10, 10],   # Position Angle [deg]
        [-5, 5],    # Offset RA [arcsec]
        [-5, 5],    # Offset Dec [arcsec]
        [-5, 15],   # Blob 1 Peak [log(Jy/sr)]
        [0, 1],   # Blob 1 Width [arcsec]
        [5, 10],     # Blob 1 Dist [arcsec]
        [150, 200], # Blob 1 Angle [deg]
        [-5, 15],   # Blob 2 Peak [log(Jy/sr)]
        [0, 1],   # Blob 2 Width [arcsec]
        [5, 10],     # Blob 2 Dist [arcsec]
        [330, 380], # Blob 2 Angle [deg]
        [-5, 15],   # Blob 3 Peak [log(Jy/sr)]
        [0, 1],   # Blob 3 Width [arcsec]
        [5, 10],     # Blob 3 Dist [arcsec]
        [140, 190]  # Blob 3 Angle [deg]
    ], dtype=float)

    low = prior_ranges[:, 0]
    high = prior_ranges[:, 1]

    v = low + u * (high - low)
    
    return v
