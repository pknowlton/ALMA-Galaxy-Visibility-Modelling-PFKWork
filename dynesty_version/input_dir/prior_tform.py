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
### 2D Gaussian Ring + 1 Gaussian Blob (2 Peaks)
#########################

def twod_gauss1blob_2peak_ptform(u):
    """
    Prior transform for the 2D Gaussian Ring + 1 Gaussian Blob with 2 Peaks (13 parameters).

    Transforms unit hypercube coordinates $u \in [0, 1]^{13}$ into physical parameters
    for a ring profile superimposed with a single blob composed of two concentric Gaussian
    components with distinct peaks and widths, where Width 2 is constrained to be narrower than Width 1.

    Parameters
    ----------
    u : array_like, shape (13,)
        Normalized coordinates on the unit hypercube ($u_i \in [0, 1]$).
        Order of parameters:
            0-6:  Ring parameters (Peak, Width, Radius, Inc, PA, dRA, dDec)
            7:    Blob Peak 1 log10(Jy/sr)
            8:    Blob Width 1 (arcsec)
            9:    Blob Peak 2 log10(Jy/sr)
            10:   Blob Width 2 (arcsec, narrower than Width 1)
            11:   Blob radial distance from ring center (arcsec)
            12:   Blob azimuthal position angle (degrees)

    Returns
    -------
    v : numpy.ndarray, shape (13,)
        Physical model parameters. Prior ranges:
            - Ring Peak:       log(Jy/sr)
            - Ring Width:      arcsec      
            - Ring Radius:     arcsec      
            - Inclination:     degrees     
            - Position Angle:  degrees     
            - Offset RA:       arcsec      
            - Offset Dec:      arcsec      
            - Blob Peak 1:     log(Jy/sr)  
            - Blob Width 1:    arcsec      
            - Blob Peak 2:     log(Jy/sr)  
            - Blob Width 2:    arcsec      
            - Blob Distance:   arcsec      
            - Blob Angle:      degrees     
    """
    u = np.asarray(u)
    prior_ranges = np.array([
        [0, 8],   # Ring Peak [log(Jy/sr)]
        [0, 3],    # Ring Width [arcsec]
        [4, 8],    # Ring Rad [arcsec]
        [45, 80],    # Inclination [deg]
        [-10, 10],  # Position Angle [deg]
        [-5, 5],    # Offset RA [arcsec]
        [-5, 5],    # Offset Dec [arcsec]
        [0, 9],   # Blob Peak 1 [log(Jy/sr)]
        #[0.3, 1.0],   # Blob Width 1 [arcsec] (Blob 1)
        #[0.3, 1.0],   # Blob Width 1 [arcsec] (Blob 2)
        [0.2, 1.0],   # Blob Width 1 [arcsec] (Blob 3)
        [0, 9],   # Blob Peak 2 [log(Jy/sr)]
        #[0, 0.3],   # Blob Width 2 [arcsec] (Blob 1)
        #[0, 0.3],   # Blob Width 2 [arcsec] (Blob 2)
        [0, 0.2],   # Blob Width 2 [arcsec] (Blob 3)
        [5, 9],     # Blob Dist [arcsec]
        #[168, 200]  # Blob Angle [deg] (Blob 1)
        #[350, 390]  # Blob Angle [deg] (Blob 2)
        [150, 168]  # Blob Angle [deg] (Blob 3)
    ], dtype=float)

    low = prior_ranges[:, 0]
    high = prior_ranges[:, 1]

    v = low + u * (high - low)
    
    # Constrain Width 2 to always be narrower than Width 1
    #v[10] = prior_ranges[10, 0] + u[10] * (v[8] - prior_ranges[10, 0])
    
    return v


#########################
### 2D Gaussian Ring + 1 Gaussian Blob (2 Peaks, Double Pendulum)
#########################

def twod_gauss1blob_2peak_dp_ptform(u):
    """
    Prior transform for the 2D Gaussian Ring + 1 Gaussian Blob with 2 Peaks (Double Pendulum, 15 parameters).

    Transforms unit hypercube coordinates $u \in [0, 1]^{15}$ into physical parameters
    for a ring profile superimposed with a single clump having two separate emission peaks.
    Peak 2 is parameterized relative to Peak 1: its distance from Peak 1 is constrained to $\le 2$ arcseconds,
    and its angle relative to Peak 1 covers $[0, 360]^\circ$.

    Parameters
    ----------
    u : array_like, shape (15,)
        Normalized coordinates on the unit hypercube ($u_i \in [0, 1]$).
        Order of parameters:
            0-6:  Ring parameters (Peak, Width, Radius, Inc, PA, dRA, dDec)
            7:    Blob Peak 1 log10(Jy/sr)
            8:    Blob Width 1 (arcsec)
            9:    Blob Dist 1 (arcsec, distance of Peak 1 from ring center)
            10:   Blob Angle 1 (degrees, position angle of Peak 1 from ring center)
            11:   Blob Peak 2 log10(Jy/sr)
            12:   Blob Width 2 (arcsec)
            13:   Blob Dist 2 (arcsec, separation of Peak 2 from Peak 1, max 2 arcsec)
            14:   Blob Angle 2 (degrees, position angle of Peak 2 relative to Peak 1)

    Returns
    -------
    v : numpy.ndarray, shape (15,)
        Physical model parameters. Prior ranges:
            - Ring Peak:       log(Jy/sr)
            - Ring Width:      arcsec
            - Ring Radius:     arcsec
            - Inclination:     degrees
            - Position Angle:  degrees
            - Offset RA:       arcsec
            - Offset Dec:      arcsec
            - Blob Peak 1:     log(Jy/sr)
            - Blob Width 1:    arcsec
            - Blob Dist 1:     arcsec
            - Blob Angle 1:    degrees
            - Blob Peak 2:     log(Jy/sr)
            - Blob Width 2:    arcsec
            - Blob Dist 2:     arcsec (0 to 2 arcsec from Peak 1)
            - Blob Angle 2:    degrees (0 to 360 deg from Peak 1)
    """
    u = np.asarray(u)
    prior_ranges = np.array([
        [0, 8],   # Ring Peak [log(Jy/sr)]
        [0, 3],    # Ring Width [arcsec]
        [4, 8],    # Ring Rad [arcsec]
        [45, 80],    # Inclination [deg]
        [-10, 10],  # Position Angle [deg]
        [-5, 5],    # Offset RA [arcsec]
        [-5, 5],    # Offset Dec [arcsec]
        [0, 9],   # Blob Peak 1 [log(Jy/sr)]
        [0.0, 1.0],   # Blob Width 1 [arcsec] (Blob 1)
        #[0.0, 1.0],   # Blob Width 1 [arcsec] (Blob 2)
        #[0.0, 1.0],   # Blob Width 1 [arcsec] (Blob 3)
        [5, 9],     # Blob Dist 1 [arcsec]
        [168, 200],  # Blob Angle 1 [deg] (Blob 1)
        #[350, 390]  # Blob Angle 1 [deg] (Blob 2)
        #[150, 168]  # Blob Angle 1 [deg] (Blob 3)
        [0, 9],   # Blob Peak 2 [log(Jy/sr)]
        [0.0, 1.0],   # Blob Width 2 [arcsec] (Blob 1)
        #[0.0, 1.0],   # Blob Width 2 [arcsec] (Blob 2)
        #[0.0, 1.0],   # Blob Width 2 [arcsec] (Blob 3)
        [0, 2],     # Blob Dist 2 [arcsec] - distance from peak 1
        [0, 360]  # Blob Angle 2 [deg] - angle from peak 1
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
        [0, 0.6],   # Blob 1 Width [arcsec]
        [5, 10],     # Blob 1 Dist [arcsec]
        [150, 200], # Blob 1 Angle [deg]
        [-5, 15],   # Blob 2 Peak [log(Jy/sr)]
        [0, 0.6],   # Blob 2 Width [arcsec]
        [5, 10],     # Blob 2 Dist [arcsec]
        [330, 380], # Blob 2 Angle [deg]
        [-5, 15],   # Blob 3 Peak [log(Jy/sr)]
        [0, 0.6],   # Blob 3 Width [arcsec]
        [5, 10],     # Blob 3 Dist [arcsec]
        [140, 190]  # Blob 3 Angle [deg]
    ], dtype=float)

    low = prior_ranges[:, 0]
    high = prior_ranges[:, 1]

    v = low + u * (high - low)
    
    return v
