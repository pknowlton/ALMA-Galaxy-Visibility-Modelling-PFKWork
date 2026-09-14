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

Prior Design & Harmonization:
-----------------------------
1. **Shared-Parameter Prior Consistency**:
   All models (`twod_gaussring`, `twod_gauss1blob`, `twod_gauss1blob_2peak`,
   `twod_gauss1blob_2peak_dp`, `twod_gauss3blob`) share the exact same prior boundaries
   for common disk parameters:
     - Ring Peak:        [0, 10] log10(Jy/sr)
     - Ring Sigma:       [0.05, 4.0] arcsec (strictly positive lower bound >= 0.05")
     - Ring Radius:      [3.0, 10.0] arcsec
     - Disk Inclination: [0, 85] deg
     - Position Angle:   [0, 180] deg (180 deg axial periodicity for symmetric ring)
     - Centroid dRA:     [-4.0, 4.0] arcsec
     - Centroid dDec:    [-4.0, 4.0] arcsec
   This harmonization is mandatory for meaningful Bayesian evidence comparisons ($\ln\mathcal{Z}$)
   and Bayes factors between competing model complexities.

2. **Strictly Positive Resolution-Aware Width Bounds**:
   All Gaussian standard deviations ($\sigma$) have a strictly positive lower bound of
   $\ge 0.05$ arcsec (corresponding to $\ge 3$ pixels across FWHM on a 0.039" grid).
   This eliminates the zero-width funnel degeneracy, prevents division-by-zero singularities,
   and avoids underflow/aliasing artifacts.

3. **Label Symmetry Breaking**:
   In multi-component models (e.g. `twod_gauss3blob`), non-overlapping azimuthal prior
   ranges are assigned to distinct star-forming clumps (matching Sun et al. 2024 YMCs),
   preventing unconstrained label switching and multi-modal posterior degeneracies.
"""

import numpy as np

# Standardized shared ring prior bounds across ALL model profiles
# Audit P1 Item 1: Reparameterized as log integrated flux (Jy) + log size (arcsec)
RING_PRIOR_RANGES = np.array([
    [-3.0, 0.0],     # Ring LogFlux [log10(Jy)] (1 mJy to 1 Jy)
    [-1.301, 0.602], # Ring LogSigma [log10(arcsec)] (0.05" to 4.0", strictly positive)
    [3.0, 10.0],     # Ring Radius [arcsec]
    [0.0, 85.0],     # Inclination [deg]
    [0.0, 180.0],    # Position Angle [deg] (axial 180 deg interval)
    [-4.0, 4.0],     # Offset RA [arcsec]
    [-4.0, 4.0]      # Offset Dec [arcsec]
], dtype=float)

def uniform_area_radius(u, r_min, r_max):
    """
    Transforms uniform hypercube u in [0, 1] to radial distance r with uniform area density.
    p(r) dr proportional to r dr  =>  r = sqrt(r_min^2 + u * (r_max^2 - r_min^2))
    Audit P1 Item 2: Uniform area Jacobian on disk; never includes r = 0.
    """
    return np.sqrt(r_min**2 + u * (r_max**2 - r_min**2))

#########################
### 2D Gaussian Ring (7 parameters)
#########################

def twod_gaussring_ptform(u):
    """
    Prior transform for the 2D Gaussian Ring model (7 parameters).
    Reparameterized as log integrated flux + log size (Audit P1 Item 1).

    Parameters
    ----------
    u : array_like, shape (7,)
        Unit hypercube coordinates in [0, 1].

    Returns
    -------
    v : numpy.ndarray, shape (7,)
        Physical model parameters:
        [Ring LogFlux (Jy), Ring LogSigma (arcsec), Ring Rad (arcsec),
         Inc (deg), PA (deg), Offset RA (arcsec), Offset Dec (arcsec)]
    """
    u = np.asarray(u)
    low = RING_PRIOR_RANGES[:, 0]
    high = RING_PRIOR_RANGES[:, 1]
    return low + u * (high - low)

#########################
### 2D Gaussian Ring + 1 Gaussian Blob (11 parameters)
#########################

def twod_gauss1blob_ptform(u):
    """
    Prior transform for 2D Gaussian Ring + 1 Gaussian Blob (11 parameters).
    Reparameterized as log integrated flux + log size (Audit P1 Item 1).

    Parameters
    ----------
    u : array_like, shape (11,)
        Unit hypercube coordinates in [0, 1].

    Returns
    -------
    v : numpy.ndarray, shape (11,)
        Physical model parameters:
        0-6:  Ring parameters (shared with twod_gaussring)
        7:    Blob 1 LogFlux [log10(Jy)]
        8:    Blob 1 LogSigma [log10(arcsec)]
        9:    Blob 1 Distance [arcsec] (uniform area on disk)
        10:   Blob 1 Angle [deg] (East of North in disk frame)
    """
    u = np.asarray(u)
    v = np.zeros(11, dtype=float)
    # Shared ring parameters
    low_ring = RING_PRIOR_RANGES[:, 0]
    high_ring = RING_PRIOR_RANGES[:, 1]
    v[:7] = low_ring + u[:7] * (high_ring - low_ring)

    # Blob 1: LogFlux in [-5.5, -3.0], LogSigma in [-1.301, 0.0] (0.05" to 1.0")
    v[7] = -5.5 + u[7] * (-3.0 - (-5.5))
    v[8] = -1.301 + u[8] * (0.0 - (-1.301))
    # Distance: uniform area density on disk in [4.0, 10.0] arcsec (Audit P1 Item 2)
    v[9] = uniform_area_radius(u[9], 4.0, 10.0)
    # Angle: South cluster region [150, 210] deg
    v[10] = 150.0 + u[10] * (210.0 - 150.0)
    return v

#########################
### 2D Gaussian Ring + 1 Gaussian Blob (2 Concentric Peaks, 13 parameters)
#########################

def twod_gauss1blob_2peak_ptform(u):
    """
    Prior transform for 2D Gaussian Ring + 1 Gaussian Blob with 2 Peaks (13 parameters).
    Reparameterized as log integrated flux + log size (Audit P1 Item 1).

    Models a single emission knot as a core + envelope concentric Gaussian structure.
    Peak 1 (Core):     LogSigma in [-1.301, -0.398] (0.05" to 0.4")
    Peak 2 (Envelope): LogSigma in [-0.398, 0.176]  (0.4" to 1.5")

    Parameters
    ----------
    u : array_like, shape (13,)
        Unit hypercube coordinates in [0, 1].

    Returns
    -------
    v : numpy.ndarray, shape (13,)
        Physical model parameters:
        0-6:  Ring parameters (shared)
        7:    Blob LogFlux 1 (Core) [log10(Jy)]
        8:    Blob LogSigma 1 (Core) [log10(arcsec)]
        9:    Blob LogFlux 2 (Envelope) [log10(Jy)]
        10:   Blob LogSigma 2 (Envelope) [log10(arcsec)]
        11:   Blob Distance [arcsec] (uniform area on disk)
        12:   Blob Angle [deg]
    """
    u = np.asarray(u)
    v = np.zeros(13, dtype=float)
    low_ring = RING_PRIOR_RANGES[:, 0]
    high_ring = RING_PRIOR_RANGES[:, 1]
    v[:7] = low_ring + u[:7] * (high_ring - low_ring)

    # Core (Peak 1)
    v[7] = -5.5 + u[7] * (-3.0 - (-5.5))
    v[8] = -1.301 + u[8] * (-0.398 - (-1.301))
    # Envelope (Peak 2)
    v[9] = -5.5 + u[9] * (-3.0 - (-5.5))
    v[10] = -0.398 + u[10] * (0.176 - (-0.398))
    # Position
    v[11] = uniform_area_radius(u[11], 4.0, 10.0)
    v[12] = 150.0 + u[12] * (210.0 - 150.0)
    return v

#########################
### 2D Gaussian Ring + 1 Gaussian Blob (2 Peaks, Double Pendulum, 15 parameters)
#########################

def twod_gauss1blob_2peak_dp_ptform(u):
    """
    Prior transform for 2D Gaussian Ring + 1 Gaussian Blob with 2 Peaks (Double Pendulum, 15 parameters).
    Reparameterized as log integrated flux + log size (Audit P1 Item 1).

    Peak 1 is located at (Dist 1, Angle 1) relative to galaxy center.
    Peak 2 is located at (Dist 2, Angle 2) relative to Peak 1, with Dist 2 in [0.05, 2.0] arcsec.
    The strictly positive lower bound of 0.05" on Dist 2 prevents Peak 2 from collapsing
    into Peak 1 (which would make Angle 2 undefined and cause coordinate singularity).

    Parameters
    ----------
    u : array_like, shape (15,)
        Unit hypercube coordinates in [0, 1].

    Returns
    -------
    v : numpy.ndarray, shape (15,)
        Physical model parameters:
        0-6:  Ring parameters (shared)
        7:    Blob LogFlux 1 [log10(Jy)]
        8:    Blob LogSigma 1 [log10(arcsec)]
        9:    Blob Dist 1 [arcsec] (uniform area on disk)
        10:   Blob Angle 1 [deg]
        11:   Blob LogFlux 2 [log10(Jy)]
        12:   Blob LogSigma 2 [log10(arcsec)]
        13:   Blob Dist 2 [arcsec] (separation from Peak 1: 0.05" to 2.0", uniform area)
        14:   Blob Angle 2 [deg] (position angle relative to Peak 1: 0 to 360 deg)
    """
    u = np.asarray(u)
    v = np.zeros(15, dtype=float)
    low_ring = RING_PRIOR_RANGES[:, 0]
    high_ring = RING_PRIOR_RANGES[:, 1]
    v[:7] = low_ring + u[:7] * (high_ring - low_ring)

    # Peak 1
    v[7] = -5.5 + u[7] * (-3.0 - (-5.5))
    v[8] = -1.301 + u[8] * (0.0 - (-1.301))
    v[9] = uniform_area_radius(u[9], 4.0, 10.0)
    v[10] = 330.0 + u[10] * (390.0 - 330.0)

    # Peak 2 (Double Pendulum relative offset)
    v[11] = -5.5 + u[11] * (-3.0 - (-5.5))
    v[12] = -1.301 + u[12] * (0.0 - (-1.301))
    v[13] = uniform_area_radius(u[13], 0.05, 2.0)
    v[14] = 0.0 + u[14] * 360.0
    return v

#########################
### 2D Gaussian Ring + 3 Gaussian Blobs (19 parameters)
#########################

def twod_gauss3blob_ptform(u):
    """
    Prior transform for 2D Gaussian Ring + 3 Gaussian Blobs (19 parameters).
    Reparameterized as log integrated flux + log size (Audit P1 Item 1).

    Assigns distinct, non-overlapping angular regions to Blobs 1, 2, and 3
    under the astronomical counter-clockwise (East of North) convention:
    - Blob 1: South-East clump (YMC 15 complex, disk angle ~171 deg): Angle in [155, 177] deg.
    - Blob 2: North clump (YMC 6 complex, disk angle ~348 deg): Angle in [330, 375] deg.
    - Blob 3: South-West clump (YMC 17/18 complex, disk angle ~182-185 deg): Angle in [177, 205] deg.

    This deliberate symmetry breaking eliminates unconstrained label switching
    and resolves multimodal posterior degeneracies between the two southern clumps.

    Parameters
    ----------
    u : array_like, shape (19,)
        Unit hypercube coordinates in [0, 1].

    Returns
    -------
    v : numpy.ndarray, shape (19,)
        Physical model parameters:
        0-6:   Ring parameters (shared)
        7-10:  Blob 1 (LogFlux, LogSigma, Dist, Angle)
        11-14: Blob 2 (LogFlux, LogSigma, Dist, Angle)
        15-18: Blob 3 (LogFlux, LogSigma, Dist, Angle)
    """
    u = np.asarray(u)
    v = np.zeros(19, dtype=float)
    low_ring = RING_PRIOR_RANGES[:, 0]
    high_ring = RING_PRIOR_RANGES[:, 1]
    v[:7] = low_ring + u[:7] * (high_ring - low_ring)

    # Blob 1 (YMC 15 complex, South-East)
    v[7] = -5.5 + u[7] * (-3.0 - (-5.5))
    v[8] = -1.301 + u[8] * (0.0 - (-1.301))
    v[9] = uniform_area_radius(u[9], 4.0, 10.0)
    v[10] = 155.0 + u[10] * (177.0 - 155.0)

    # Blob 2 (YMC 6 complex, North)
    v[11] = -5.5 + u[11] * (-3.0 - (-5.5))
    v[12] = -1.301 + u[12] * (0.0 - (-1.301))
    v[13] = uniform_area_radius(u[13], 4.0, 10.0)
    v[14] = 330.0 + u[14] * (375.0 - 330.0)

    # Blob 3 (YMC 17/18 complex, South-West)
    v[15] = -5.5 + u[15] * (-3.0 - (-5.5))
    v[16] = -1.301 + u[16] * (0.0 - (-1.301))
    v[17] = uniform_area_radius(u[17], 4.0, 10.0)
    v[18] = 177.0 + u[18] * (205.0 - 177.0)
    return v
