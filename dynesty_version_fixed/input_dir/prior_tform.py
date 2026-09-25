r"""
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

    # Blob 1: LogFlux in [-5.5, -3.0], LogSigma in [-1.301, -0.398] (0.05" to 0.4")
    v[7] = -5.5 + u[7] * (-3.0 - (-5.5))
    v[8] = -1.301 + u[8] * (-0.398 - (-1.301))
    # Distance: uniform area density on disk in [4.0, 10.0] arcsec (Audit P1 Item 2)
    v[9] = uniform_area_radius(u[9], 4.0, 10.0)
    # Angle: North cluster region [330.0, 390.0] deg
    v[10] = 330.0 + u[10] * (390.0 - 330.0)
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
    v[12] = 330.0 + u[12] * (390.0 - 330.0)
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
### 2D Gaussian Ring + 2 Gaussian Blobs (15 parameters)
#########################

def twod_gauss2blob_ptform(u):
    """
    Prior transform for 2D Gaussian Ring + 2 Gaussian Blobs (15 parameters).
    Reparameterized as log integrated flux + log size (Audit P1 Item 1).

    Parameters
    ----------
    u : array_like, shape (15,)
        Unit hypercube coordinates in [0, 1].

    Returns
    -------
    v : numpy.ndarray, shape (15,)
        Physical model parameters:
        0-6:   Ring parameters (shared)
        7-10:  Blob 1 (LogFlux, LogSigma, Dist, Angle) [North clump: 330 to 390 deg]
        11-14: Blob 2 (LogFlux, LogSigma, Dist, Angle) [South-West clump: 177 to 205 deg]
    """
    u = np.asarray(u)
    v = np.zeros(15, dtype=float)
    low_ring = RING_PRIOR_RANGES[:, 0]
    high_ring = RING_PRIOR_RANGES[:, 1]
    v[:7] = low_ring + u[:7] * (high_ring - low_ring)

    # Blob 1 (North clump)
    v[7] = -5.5 + u[7] * (-3.0 - (-5.5))
    v[8] = -1.301 + u[8] * (-0.398 - (-1.301))
    v[9] = uniform_area_radius(u[9], 4.0, 10.0)
    v[10] = 330.0 + u[10] * (390.0 - 330.0)

    # Blob 2 (South-West clump)
    v[11] = -5.5 + u[11] * (-3.0 - (-5.5))
    v[12] = -1.301 + u[12] * (-0.398 - (-1.301))
    v[13] = uniform_area_radius(u[13], 4.0, 10.0)
    v[14] = 177.0 + u[14] * (205.0 - 177.0)
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
    - Blob 1: North clump (YMC 6 complex, disk angle ~348 deg): Angle in [330, 390] deg.
    - Blob 2: South-West clump (YMC 17/18 complex, disk angle ~182-185 deg): Angle in [177, 205] deg.
    - Blob 3: South-East clump (YMC 15 complex, disk angle ~171 deg): Angle in [155, 177] deg.

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

    # Blob 1: North clump
    v[7] = -5.5 + u[7] * (-3.0 - (-5.5))
    v[8] = -1.301 + u[8] * (-0.398 - (-1.301))
    v[9] = uniform_area_radius(u[9], 4.0, 10.0)
    v[10] = 330.0 + u[10] * (390.0 - 330.0)

    # Blob 2: South-West clump
    v[11] = -5.5 + u[11] * (-3.0 - (-5.5))
    v[12] = -1.301 + u[12] * (-0.398 - (-1.301))
    v[13] = uniform_area_radius(u[13], 4.0, 10.0)
    v[14] = 177.0 + u[14] * (205.0 - 177.0)

    # Blob 3: South-East clump
    v[15] = -5.5 + u[15] * (-3.0 - (-5.5))
    v[16] = -1.301 + u[16] * (-0.398 - (-1.301))
    v[17] = uniform_area_radius(u[17], 4.0, 10.0)
    v[18] = 155.0 + u[18] * (177.0 - 155.0)
    return v

#########################
### Simulated Single Gaussian Blob (7 parameters)
#########################

def simgauss_ptform(u):
    """
    Prior transform for the simulated single Gaussian blob test model (7 parameters).
    Reparameterized according to Audit P1 Item 1 (log integrated flux + log size)
    and Audit P1 Item 2 (uniform area Jacobian on disk).

    Physical model parameters:
        Peak brightness: 10^8 to 10^10 Jy/sr
        Gaussian width sigma: 0.6 to 1.4 arcsec (FWHM = 2.355 arcsec at sigma = 1.0 arcsec)
        Integrated flux: F = 2 * pi * sigma^2 * I_0
                         log10(F [Jy]) spans [-2.27, 0.46] for I_0 in [10^8, 10^10] Jy/sr
                         and sigma in [0.6, 1.4] arcsec (centered at [-1.83, 0.17] for sigma = 1.0 arcsec).
        Azimuthal angle: 90 to 180 deg counter-clockwise (East of North)
        Radial distance: 5 to 7 arcsec (uniform area on disk)
        Position angle: 0 to 15 deg
        Centroid offsets (dRA, dDec): -4 to 4 arcsec

    Parameters
    ----------
    u : array_like, shape (7,)
        Unit hypercube coordinates in [0, 1].

    Returns
    -------
    v : numpy.ndarray, shape (7,)
        0: Blob LogFlux [log10(Jy)] in [-2.27, 0.46]
        1: Blob LogSigma [log10(arcsec)] in [-0.222, 0.146] (0.6" to 1.4")
        2: Blob Distance [arcsec] in [5.0, 7.0] (uniform area on disk)
        3: Blob Angle [deg] in [90.0, 180.0] (East of North, CCW)
        4: Position Angle [deg] in [0.0, 15.0]
        5: Centroid Offset RA [arcsec] in [-4.0, 4.0]
        6: Centroid Offset Dec [arcsec] in [-4.0, 4.0]
    """
    u = np.asarray(u)
    v = np.zeros(7, dtype=float)

    # Blob LogFlux [log10(Jy)] derived from peak 10^8 - 10^10 Jy/sr and sigma 0.6 - 1.4"
    v[0] = -2.27 + u[0] * (0.46 - (-2.27))
    # Blob LogSigma [log10(arcsec)] corresponding to sigma in [0.6, 1.4]"
    v[1] = -0.222 + u[1] * (0.146 - (-0.222))
    # Distance: uniform area density on disk in [5.0, 7.0] arcsec (Audit P1 Item 2)
    v[2] = uniform_area_radius(u[2], 5.0, 7.0)
    # Angle: [90.0, 180.0] deg (counter-clockwise East of North)
    v[3] = 90.0 + u[3] * (180.0 - 90.0)
    # Disk Position Angle: [0.0, 15.0] deg
    v[4] = 0.0 + u[4] * (15.0 - 0.0)
    # Centroid Offset RA: [-4.0, 4.0] arcsec
    v[5] = -4.0 + u[5] * (4.0 - (-4.0))
    # Centroid Offset Dec: [-4.0, 4.0] arcsec
    v[6] = -4.0 + u[6] * (4.0 - (-4.0))
    return v

twod_simgauss_ptform = simgauss_ptform

