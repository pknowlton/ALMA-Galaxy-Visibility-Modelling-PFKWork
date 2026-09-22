r"""
Prior Transformations for Dynesty Nested Sampling (Streamlined Edition)
========================================================================
In Bayesian nested sampling (Dynesty), sampling is performed on an N-dimensional
unit hypercube where coordinates $u_i \in [0, 1]$ represent cumulative probability
values. A prior transform function $\theta = F^{-1}(u)$ maps these coordinates
to physical parameter values.

Key Architecture & User Ergonomics:
-----------------------------------
1. Intuitive Units for Easy Editing:
   You can directly edit prior boundaries in familiar physical units:
     - Surface Brightness:  log10(Jy/sr)
     - Gaussian Widths:     arcseconds (strictly positive, e.g. >= 0.05")
     - Spatial Positions:   arcseconds (ring radius, centroid offsets, clump separations)
     - Angular Coordinates: degrees (disk inclination, position angles, azimuthal angles)

2. Pre-Sampling Compilation (Zero Runtime Overhead):
   Interferometric visibilities directly constrain total integrated flux F = V(0).
   Sampling in (log10 Flux, log10 Sigma) eliminates the curved flux-size "funnel"
   degeneracy that plagues (peak, sigma) parameterizations.

   To avoid computing unit conversions repeatedly during sampling, ALL intuitive
   prior ranges are converted ONCE at module load / import time into compiled
   Dynesty ranges:
     - Peak [log10(Jy/sr)] -> LogFlux [log10(Jy)]
     - Sigma [arcsec]      -> LogSigma [log10(arcsec)]
   Dynesty calls `ptform(u)` millions of times; during sampling, only ultra-fast
   linear interpolation and uniform-area disk transforms are executed. Zero conversion
   overhead occurs during the nested sampling run!

3. Shared Common Ring Priors:
   All models inherit identical ring prior boundaries from `COMMON_RING_PRIORS`.
   Modifying `COMMON_RING_PRIORS` automatically propagates to all model profiles,
   preserving Bayesian evidence comparability and Bayes factors across models.

4. Familiar Table Formatting:
   The user configuration section is styled cleanly and intuitively, matching
   the readable table format of the original `prior_tform.py`.
"""

import numpy as np

# Physical conversion constant: radians per arcsecond
ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)


# =============================================================================
# CONVERSION HELPER FUNCTIONS (Executed ONCE Ahead of Sampling)
# =============================================================================

def sigma_to_logsigma(sigma_arcsec):
    """
    Converts Gaussian standard deviation sigma in arcseconds to log10(sigma [arcsec]).
    """
    sigma = float(sigma_arcsec)
    if sigma <= 0.0:
        raise ValueError(f"Gaussian sigma must be strictly positive (> 0 arcsec), got {sigma}")
    return float(np.round(np.log10(sigma), 3))


def blob_peak_to_logflux(peak_jysr, sigma_bounds=None, sigma_arcsec=None):
    """
    Converts Gaussian blob peak surface brightness log10(Jy/sr) to total integrated flux log10(Jy).

    Derivation (Audit P1 Item 1):
        F_blob = 2 * pi * I_0 * sigma_rad^2
        log10(F_blob) = peak + log10(2 * pi * sigma_rad^2)
    """
    if sigma_arcsec is not None:
        s_ref = float(sigma_arcsec)
    elif sigma_bounds is not None:
        s_ref = float(np.sqrt(sigma_bounds[0] * sigma_bounds[1]))
    else:
        s_ref = 0.22360679774997896  # Geometric mean of default [0.05, 1.0] arcsec

    sigma_rad = s_ref * ARCSEC_TO_RAD
    area_sr = 2.0 * np.pi * (sigma_rad**2)
    return float(np.round(peak_jysr + np.log10(area_sr), 3))


def ring_peak_to_logflux(peak_jysr, rad_bounds=None, sigma_bounds=None, inc_deg=65.0):
    """
    Converts inclined Gaussian ring peak brightness log10(Jy/sr) to total integrated flux log10(Jy).

    Derivation (Audit P1 Item 1):
        F_ring = (2 * pi)^(3/2) * I_0 * R_rad * sigma_rad * cos(i)
        log10(F_ring) = peak + log10((2*pi)^(3/2) * R_rad * sigma_rad * cos(i))
    """
    r_ref = float(np.sqrt(rad_bounds[0] * rad_bounds[1])) if rad_bounds is not None else 5.477225575051661
    s_ref = float(np.sqrt(sigma_bounds[0] * sigma_bounds[1])) if sigma_bounds is not None else 0.4472135954999579
    inc_rad = np.radians(float(inc_deg))

    area_sr = ((2.0 * np.pi)**1.5) * (r_ref * ARCSEC_TO_RAD) * (s_ref * ARCSEC_TO_RAD) * np.cos(inc_rad)
    return float(np.round(peak_jysr + np.log10(area_sr), 3))


def uniform_area_radius(u, r_min, r_max):
    """
    Transforms uniform hypercube coordinate u in [0, 1] to radial distance r with uniform area density.
    p(r) dr proportional to r dr  =>  r = sqrt(r_min^2 + u * (r_max^2 - r_min^2))
    Audit P1 Item 2: Non-singular uniform area Jacobian on disk.
    """
    return np.sqrt(r_min**2 + u * (r_max**2 - r_min**2))


# =============================================================================
# USER CONFIGURABLE PRIOR RANGES (Intuitive Units: log10(Jy/sr) and arcsec)
# =============================================================================
# You can directly edit the values in this section!
# When this script is loaded, these intuitive ranges are automatically compiled
# into the (log_flux, log_sigma) parameter space sampled by Dynesty.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. SHARED COMMON RING PRIOR RANGES (Shared across ALL galaxy models)
# -----------------------------------------------------------------------------
COMMON_RING_PRIORS = np.array([
    [6.417, 9.417], # 0: Ring Peak brightness [log10(Jy/sr)] -> converted to LogFlux [-3.0, 0.0] Jy
    [0.05, 4.0],    # 1: Ring Gaussian width sigma [arcsec]  -> converted to LogSigma [-1.301, 0.602]
    [3.0, 10.0],    # 2: Ring center radius [arcsec]
    [0.0, 85.0],    # 3: Disk inclination angle [deg]
    [0.0, 180.0],   # 4: Disk position angle PA [deg] (East of North, 180 deg periodicity)
    [-4.0, 4.0],    # 5: Centroid RA offset dRA [arcsec]
    [-4.0, 4.0]     # 6: Centroid Dec offset dDec [arcsec]
], dtype=float)

# -----------------------------------------------------------------------------
# 2. MODEL-SPECIFIC BLOB / CLUMP PRIOR RANGES
# -----------------------------------------------------------------------------

# twod_gauss1blob: Single Gaussian Blob (parameters 7-10)
GAUSS1BLOB_USER_PRIORS = np.array([
    [5.632, 8.132], # 7:  Blob 1 Peak [log10(Jy/sr)]   -> converted to LogFlux [-5.5, -3.0]
    [0.05, 1.0],    # 8:  Blob 1 Width sigma [arcsec]  -> converted to LogSigma [-1.301, 0.0]
    [4.0, 10.0],    # 9:  Blob 1 Radial Dist [arcsec]  (sampled uniformly in area on disk)
    [150.0, 210.0]  # 10: Blob 1 Azimuthal Angle [deg] (East of North, counter-clockwise)
], dtype=float)

# twod_gauss1blob_2peak: Concentric Core + Envelope Clump (parameters 7-12)
GAUSS1BLOB_2PEAK_USER_PRIORS = np.array([
    [6.030, 8.530], # 7:  Blob Core Peak [log10(Jy/sr)]      -> converted to LogFlux [-5.5, -3.0]
    [0.05, 0.4],    # 8:  Blob Core Width sigma [arcsec]     -> converted to LogSigma [-1.301, -0.398]
    [4.553, 7.053], # 9:  Blob Envelope Peak [log10(Jy/sr)]  -> converted to LogFlux [-5.5, -3.0]
    [0.4, 1.5],     # 10: Blob Envelope Width sigma [arcsec] -> converted to LogSigma [-0.398, 0.176]
    [4.0, 10.0],    # 11: Blob Radial Dist [arcsec]          (sampled uniformly in area on disk)
    [330.0, 390.0]  # 12: Blob Azimuthal Angle [deg]         (East of North, counter-clockwise)
], dtype=float)

# twod_gauss1blob_2peak_dp: Double Pendulum Clump (parameters 7-14)
GAUSS1BLOB_2PEAK_DP_USER_PRIORS = np.array([
    [5.632, 8.132], # 7:  Peak 1 brightness [log10(Jy/sr)]   -> converted to LogFlux [-5.5, -3.0]
    [0.05, 1.0],    # 8:  Peak 1 Width sigma [arcsec]        -> converted to LogSigma [-1.301, 0.0]
    [4.0, 10.0],    # 9:  Peak 1 Radial Dist [arcsec]        (sampled uniformly in area on disk)
    [330.0, 390.0], # 10: Peak 1 Azimuthal Angle [deg]       (East of North, counter-clockwise)
    [5.632, 8.132], # 11: Peak 2 brightness [log10(Jy/sr)]   -> converted to LogFlux [-5.5, -3.0]
    [0.05, 1.0],    # 12: Peak 2 Width sigma [arcsec]        -> converted to LogSigma [-1.301, 0.0]
    [0.05, 2.0],    # 13: Peak 2 Separation from Peak 1 [arcsec] (uniform area, strictly positive)
    [0.0, 360.0]    # 14: Peak 2 Angle from Peak 1 [deg]     (full circular rotation)
], dtype=float)

# twod_gauss3blob: Three Distinct Clumps (parameters 7-18)
GAUSS3BLOB_USER_PRIORS = np.array([
    # Blob 1: South-East clump (YMC 15 complex, disk angle ~171 deg)
    [5.632, 8.132], # 7:  Blob 1 Peak [log10(Jy/sr)]   -> converted to LogFlux [-5.5, -3.0]
    [0.05, 1.0],    # 8:  Blob 1 Width sigma [arcsec]  -> converted to LogSigma [-1.301, 0.0]
    [4.0, 10.0],    # 9:  Blob 1 Radial Dist [arcsec]  (sampled uniformly in area on disk)
    [155.0, 177.0], # 10: Blob 1 Angle [deg] (East of North, CCW)
    # Blob 2: North clump (YMC 6 complex, disk angle ~348 deg)
    [5.632, 8.132], # 11: Blob 2 Peak [log10(Jy/sr)]   -> converted to LogFlux [-5.5, -3.0]
    [0.05, 1.0],    # 12: Blob 2 Width sigma [arcsec]  -> converted to LogSigma [-1.301, 0.0]
    [4.0, 10.0],    # 13: Blob 2 Radial Dist [arcsec]  (sampled uniformly in area on disk)
    [330.0, 375.0], # 14: Blob 2 Angle [deg] (East of North, CCW)
    # Blob 3: South-West clump (YMC 17/18 complex, disk angle ~182-185 deg)
    [5.632, 8.132], # 15: Blob 3 Peak [log10(Jy/sr)]   -> converted to LogFlux [-5.5, -3.0]
    [0.05, 1.0],    # 16: Blob 3 Width sigma [arcsec]  -> converted to LogSigma [-1.301, 0.0]
    [4.0, 10.0],    # 17: Blob 3 Radial Dist [arcsec]  (sampled uniformly in area on disk)
    [177.0, 205.0]  # 18: Blob 3 Angle [deg] (East of North, CCW)
], dtype=float)

# simgauss: Single simulated Gaussian blob test model (7 parameters)
SIMGAUSS_USER_PRIORS = np.array([
    [8.0, 10.0],    # 0: Blob Peak [log10(Jy/sr)]      -> converted to LogFlux [-2.27, 0.46]
    [0.6, 1.4],     # 1: Blob Width sigma [arcsec]     -> converted to LogSigma [-0.222, 0.146]
    [5.0, 7.0],     # 2: Blob Radial Dist [arcsec]     (sampled uniformly in area on disk)
    [90.0, 180.0],  # 3: Blob Azimuthal Angle [deg]    (East of North, CCW)
    [0.0, 15.0],    # 4: Position Angle [deg]
    [-4.0, 4.0],    # 5: RA Offset [arcsec]
    [-4.0, 4.0]     # 6: Dec Offset [arcsec]
], dtype=float)


# =============================================================================
# COMPILED DYNESTY PRIOR RANGES (Pre-computed Ahead of Sampling)
# =============================================================================

def compile_all_priors():
    """
    Compiles all user-defined intuitive priors into the (log_flux, log_sigma)
    parameter ranges sampled by Dynesty. This runs ONCE at module load time.
    """
    # 1. Shared Ring Priors
    r_comp = np.copy(COMMON_RING_PRIORS)
    r_comp[0, 0] = ring_peak_to_logflux(COMMON_RING_PRIORS[0, 0], rad_bounds=COMMON_RING_PRIORS[2], sigma_bounds=COMMON_RING_PRIORS[1])
    r_comp[0, 1] = ring_peak_to_logflux(COMMON_RING_PRIORS[0, 1], rad_bounds=COMMON_RING_PRIORS[2], sigma_bounds=COMMON_RING_PRIORS[1])
    r_comp[1, 0] = sigma_to_logsigma(COMMON_RING_PRIORS[1, 0])
    r_comp[1, 1] = sigma_to_logsigma(COMMON_RING_PRIORS[1, 1])

    # 2. twod_gauss1blob (Ring + 1 Blob)
    b1_comp = np.copy(GAUSS1BLOB_USER_PRIORS)
    b1_comp[0, 0] = blob_peak_to_logflux(GAUSS1BLOB_USER_PRIORS[0, 0], sigma_bounds=GAUSS1BLOB_USER_PRIORS[1])
    b1_comp[0, 1] = blob_peak_to_logflux(GAUSS1BLOB_USER_PRIORS[0, 1], sigma_bounds=GAUSS1BLOB_USER_PRIORS[1])
    b1_comp[1, 0] = sigma_to_logsigma(GAUSS1BLOB_USER_PRIORS[1, 0])
    b1_comp[1, 1] = sigma_to_logsigma(GAUSS1BLOB_USER_PRIORS[1, 1])
    gauss1blob_ranges = np.vstack([r_comp, b1_comp])

    # 3. twod_gauss1blob_2peak (Ring + Concentric Core/Envelope)
    b2_comp = np.copy(GAUSS1BLOB_2PEAK_USER_PRIORS)
    b2_comp[0, 0] = blob_peak_to_logflux(GAUSS1BLOB_2PEAK_USER_PRIORS[0, 0], sigma_bounds=GAUSS1BLOB_2PEAK_USER_PRIORS[1])
    b2_comp[0, 1] = blob_peak_to_logflux(GAUSS1BLOB_2PEAK_USER_PRIORS[0, 1], sigma_bounds=GAUSS1BLOB_2PEAK_USER_PRIORS[1])
    b2_comp[1, 0] = sigma_to_logsigma(GAUSS1BLOB_2PEAK_USER_PRIORS[1, 0])
    b2_comp[1, 1] = sigma_to_logsigma(GAUSS1BLOB_2PEAK_USER_PRIORS[1, 1])
    b2_comp[2, 0] = blob_peak_to_logflux(GAUSS1BLOB_2PEAK_USER_PRIORS[2, 0], sigma_bounds=GAUSS1BLOB_2PEAK_USER_PRIORS[3])
    b2_comp[2, 1] = blob_peak_to_logflux(GAUSS1BLOB_2PEAK_USER_PRIORS[2, 1], sigma_bounds=GAUSS1BLOB_2PEAK_USER_PRIORS[3])
    b2_comp[3, 0] = sigma_to_logsigma(GAUSS1BLOB_2PEAK_USER_PRIORS[3, 0])
    b2_comp[3, 1] = sigma_to_logsigma(GAUSS1BLOB_2PEAK_USER_PRIORS[3, 1])
    gauss1blob_2peak_ranges = np.vstack([r_comp, b2_comp])

    # 4. twod_gauss1blob_2peak_dp (Ring + Double Pendulum Clump)
    dp_comp = np.copy(GAUSS1BLOB_2PEAK_DP_USER_PRIORS)
    dp_comp[0, 0] = blob_peak_to_logflux(GAUSS1BLOB_2PEAK_DP_USER_PRIORS[0, 0], sigma_bounds=GAUSS1BLOB_2PEAK_DP_USER_PRIORS[1])
    dp_comp[0, 1] = blob_peak_to_logflux(GAUSS1BLOB_2PEAK_DP_USER_PRIORS[0, 1], sigma_bounds=GAUSS1BLOB_2PEAK_DP_USER_PRIORS[1])
    dp_comp[1, 0] = sigma_to_logsigma(GAUSS1BLOB_2PEAK_DP_USER_PRIORS[1, 0])
    dp_comp[1, 1] = sigma_to_logsigma(GAUSS1BLOB_2PEAK_DP_USER_PRIORS[1, 1])
    dp_comp[4, 0] = blob_peak_to_logflux(GAUSS1BLOB_2PEAK_DP_USER_PRIORS[4, 0], sigma_bounds=GAUSS1BLOB_2PEAK_DP_USER_PRIORS[5])
    dp_comp[4, 1] = blob_peak_to_logflux(GAUSS1BLOB_2PEAK_DP_USER_PRIORS[4, 1], sigma_bounds=GAUSS1BLOB_2PEAK_DP_USER_PRIORS[5])
    dp_comp[5, 0] = sigma_to_logsigma(GAUSS1BLOB_2PEAK_DP_USER_PRIORS[5, 0])
    dp_comp[5, 1] = sigma_to_logsigma(GAUSS1BLOB_2PEAK_DP_USER_PRIORS[5, 1])
    gauss1blob_2peak_dp_ranges = np.vstack([r_comp, dp_comp])

    # 5. twod_gauss3blob (Ring + 3 Clumps)
    b3_comp = np.copy(GAUSS3BLOB_USER_PRIORS)
    for k in (0, 4, 8):
        b3_comp[k, 0] = blob_peak_to_logflux(GAUSS3BLOB_USER_PRIORS[k, 0], sigma_bounds=GAUSS3BLOB_USER_PRIORS[k + 1])
        b3_comp[k, 1] = blob_peak_to_logflux(GAUSS3BLOB_USER_PRIORS[k, 1], sigma_bounds=GAUSS3BLOB_USER_PRIORS[k + 1])
        b3_comp[k + 1, 0] = sigma_to_logsigma(GAUSS3BLOB_USER_PRIORS[k + 1, 0])
        b3_comp[k + 1, 1] = sigma_to_logsigma(GAUSS3BLOB_USER_PRIORS[k + 1, 1])
    gauss3blob_ranges = np.vstack([r_comp, b3_comp])

    # 6. simgauss (Simulated Single Blob Test Model)
    sim_comp = np.copy(SIMGAUSS_USER_PRIORS)
    sim_comp[0, 0] = float(np.round(SIMGAUSS_USER_PRIORS[0, 0] + 2.0 * np.log10(SIMGAUSS_USER_PRIORS[1, 0] * ARCSEC_TO_RAD) + np.log10(2.0 * np.pi), 2))
    sim_comp[0, 1] = float(np.round(SIMGAUSS_USER_PRIORS[0, 1] + 2.0 * np.log10(SIMGAUSS_USER_PRIORS[1, 1] * ARCSEC_TO_RAD) + np.log10(2.0 * np.pi), 2))
    sim_comp[1, 0] = sigma_to_logsigma(SIMGAUSS_USER_PRIORS[1, 0])
    sim_comp[1, 1] = sigma_to_logsigma(SIMGAUSS_USER_PRIORS[1, 1])

    return (
        r_comp,
        gauss1blob_ranges,
        gauss1blob_2peak_ranges,
        gauss1blob_2peak_dp_ranges,
        gauss3blob_ranges,
        sim_comp
    )


# Compile ranges immediately upon module import
(
    RING_PRIOR_RANGES,
    GAUSS1BLOB_PRIOR_RANGES,
    GAUSS1BLOB_2PEAK_PRIOR_RANGES,
    GAUSS1BLOB_2PEAK_DP_PRIOR_RANGES,
    GAUSS3BLOB_PRIOR_RANGES,
    SIMGAUSS_PRIOR_RANGES
) = compile_all_priors()


# =============================================================================
# PRIOR TRANSFORM FUNCTIONS (Fast Sampling Execution: No Runtime Conversions)
# =============================================================================

#########################
### 2D Gaussian Ring (7 parameters)
#########################

def twod_gaussring_ptform(u):
    """
    Prior transform for 2D Gaussian Ring model (7 parameters).
    Parameters: [Ring LogFlux, Ring LogSigma, Radius, Inc, PA, dRA, dDec].
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
    Ring parameters (0-6) shared with twod_gaussring.
    Blob 1 parameters: [Blob LogFlux, Blob LogSigma, Dist, Angle].
    """
    u = np.asarray(u)
    v = np.empty(11, dtype=float)
    low = GAUSS1BLOB_PRIOR_RANGES[:, 0]
    high = GAUSS1BLOB_PRIOR_RANGES[:, 1]

    # Linear parameters: Ring (0-6) and Blob LogFlux, LogSigma (7-8)
    v[:9] = low[:9] + u[:9] * (high[:9] - low[:9])
    # Radial distance (index 9) with uniform area Jacobian on disk
    v[9] = uniform_area_radius(u[9], low[9], high[9])
    # Azimuthal angle (index 10)
    v[10] = low[10] + u[10] * (high[10] - low[10])
    return v


#########################
### 2D Gaussian Ring + 1 Gaussian Blob (2 Concentric Peaks, 13 parameters)
#########################

def twod_gauss1blob_2peak_ptform(u):
    """
    Prior transform for 2D Gaussian Ring + 1 Gaussian Blob with 2 concentric peaks (13 parameters).
    Models clump as Core (Peak 1) + Envelope (Peak 2).
    """
    u = np.asarray(u)
    v = np.empty(13, dtype=float)
    low = GAUSS1BLOB_2PEAK_PRIOR_RANGES[:, 0]
    high = GAUSS1BLOB_2PEAK_PRIOR_RANGES[:, 1]

    # Linear parameters: Ring (0-6) and Core/Envelope LogFlux & LogSigma (7-10)
    v[:11] = low[:11] + u[:11] * (high[:11] - low[:11])
    # Clump radial distance (index 11) with uniform area Jacobian
    v[11] = uniform_area_radius(u[11], low[11], high[11])
    # Clump azimuthal angle (index 12)
    v[12] = low[12] + u[12] * (high[12] - low[12])
    return v


#########################
### 2D Gaussian Ring + 1 Gaussian Blob (2 Peaks, Double Pendulum, 15 parameters)
#########################

def twod_gauss1blob_2peak_dp_ptform(u):
    """
    Prior transform for 2D Gaussian Ring + 1 Gaussian Blob (Double Pendulum, 15 parameters).
    Peak 2 is parameterized relative to Peak 1 with strictly positive separation.
    """
    u = np.asarray(u)
    v = np.empty(15, dtype=float)
    low = GAUSS1BLOB_2PEAK_DP_PRIOR_RANGES[:, 0]
    high = GAUSS1BLOB_2PEAK_DP_PRIOR_RANGES[:, 1]

    # Ring (0-6) and Peak 1 LogFlux, LogSigma (7-8)
    v[:9] = low[:9] + u[:9] * (high[:9] - low[:9])
    # Peak 1 radial distance (index 9)
    v[9] = uniform_area_radius(u[9], low[9], high[9])
    # Peak 1 Angle, Peak 2 LogFlux, Peak 2 LogSigma (indices 10, 11, 12)
    v[10:13] = low[10:13] + u[10:13] * (high[10:13] - low[10:13])
    # Peak 2 distance from Peak 1 (index 13, uniform area, strictly positive lower bound)
    v[13] = uniform_area_radius(u[13], low[13], high[13])
    # Peak 2 angle relative to Peak 1 (index 14, 0 to 360 deg)
    v[14] = low[14] + u[14] * (high[14] - low[14])
    return v


#########################
### 2D Gaussian Ring + 3 Gaussian Blobs (19 parameters)
#########################

def twod_gauss3blob_ptform(u):
    """
    Prior transform for 2D Gaussian Ring + 3 Gaussian Blobs (19 parameters).
    Assigns distinct, non-overlapping angular regions to Blobs 1, 2, and 3
    to break permutation symmetry and avoid label switching.
    """
    u = np.asarray(u)
    v = np.empty(19, dtype=float)
    low = GAUSS3BLOB_PRIOR_RANGES[:, 0]
    high = GAUSS3BLOB_PRIOR_RANGES[:, 1]

    # Ring (0-6)
    v[:7] = low[:7] + u[:7] * (high[:7] - low[:7])

    # Blob 1: South-East clump (YMC 15 complex)
    v[7:9] = low[7:9] + u[7:9] * (high[7:9] - low[7:9])
    v[9] = uniform_area_radius(u[9], low[9], high[9])
    v[10] = low[10] + u[10] * (high[10] - low[10])

    # Blob 2: North clump (YMC 6 complex)
    v[11:13] = low[11:13] + u[11:13] * (high[11:13] - low[11:13])
    v[13] = uniform_area_radius(u[13], low[13], high[13])
    v[14] = low[14] + u[14] * (high[14] - low[14])

    # Blob 3: South-West clump (YMC 17/18 complex)
    v[15:17] = low[15:17] + u[15:17] * (high[15:17] - low[15:17])
    v[17] = uniform_area_radius(u[17], low[17], high[17])
    v[18] = low[18] + u[18] * (high[18] - low[18])
    return v


#########################
### Simulated Single Gaussian Blob (7 parameters)
#########################

def simgauss_ptform(u):
    """
    Prior transform for simulated single Gaussian blob test model (7 parameters).
    """
    u = np.asarray(u)
    v = np.empty(7, dtype=float)
    low = SIMGAUSS_PRIOR_RANGES[:, 0]
    high = SIMGAUSS_PRIOR_RANGES[:, 1]

    # LogFlux and LogSigma (0, 1)
    v[0:2] = low[0:2] + u[0:2] * (high[0:2] - low[0:2])
    # Radial distance (index 2) with uniform area Jacobian
    v[2] = uniform_area_radius(u[2], low[2], high[2])
    # Angle, PA, dRA, dDec (indices 3-6)
    v[3:7] = low[3:7] + u[3:7] * (high[3:7] - low[3:7])
    return v

twod_simgauss_ptform = simgauss_ptform


# =============================================================================
# CLI INSPECTION / SUMMARY DISPLAY
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DYNESTY STREAMLINED PRIOR TRANSFORM CONFIGURATION & COMPILED RANGES")
    print("=" * 80)
    print(f"{'Parameter Description':<32} {'Intuitive User Prior':<22} {'Compiled Dynesty Prior':<24}")
    print("-" * 80)

    ring_labels = [
        ("Ring Surface Brightness", f"[{COMMON_RING_PRIORS[0,0]:.2f}, {COMMON_RING_PRIORS[0,1]:.2f}] log(Jy/sr)", f"[{RING_PRIOR_RANGES[0,0]:.2f}, {RING_PRIOR_RANGES[0,1]:.2f}] log(Jy)"),
        ("Ring Gaussian Width (sigma)", f"[{COMMON_RING_PRIORS[1,0]:.2f}, {COMMON_RING_PRIORS[1,1]:.2f}] arcsec", f"[{RING_PRIOR_RANGES[1,0]:.3f}, {RING_PRIOR_RANGES[1,1]:.3f}] log(arcsec)"),
        ("Ring Center Radius", f"[{COMMON_RING_PRIORS[2,0]:.1f}, {COMMON_RING_PRIORS[2,1]:.1f}] arcsec", f"[{RING_PRIOR_RANGES[2,0]:.1f}, {RING_PRIOR_RANGES[2,1]:.1f}] arcsec"),
        ("Disk Inclination", f"[{COMMON_RING_PRIORS[3,0]:.1f}, {COMMON_RING_PRIORS[3,1]:.1f}] deg", f"[{RING_PRIOR_RANGES[3,0]:.1f}, {RING_PRIOR_RANGES[3,1]:.1f}] deg"),
        ("Disk Position Angle (PA)", f"[{COMMON_RING_PRIORS[4,0]:.1f}, {COMMON_RING_PRIORS[4,1]:.1f}] deg", f"[{RING_PRIOR_RANGES[4,0]:.1f}, {RING_PRIOR_RANGES[4,1]:.1f}] deg"),
        ("Centroid Offset dRA", f"[{COMMON_RING_PRIORS[5,0]:.1f}, {COMMON_RING_PRIORS[5,1]:.1f}] arcsec", f"[{RING_PRIOR_RANGES[5,0]:.1f}, {RING_PRIOR_RANGES[5,1]:.1f}] arcsec"),
        ("Centroid Offset dDec", f"[{COMMON_RING_PRIORS[6,0]:.1f}, {COMMON_RING_PRIORS[6,1]:.1f}] arcsec", f"[{RING_PRIOR_RANGES[6,0]:.1f}, {RING_PRIOR_RANGES[6,1]:.1f}] arcsec"),
    ]

    print("Shared Ring Parameters (Inherited across ALL models):")
    for name, user_val, dyn_val in ring_labels:
        print(f"  {name:<30} {user_val:<22} {dyn_val:<24}")

    print("\nBlob 1 Parameters (twod_gauss1blob):")
    b1_labels = [
        ("Blob 1 Surface Brightness", f"[{GAUSS1BLOB_USER_PRIORS[0,0]:.2f}, {GAUSS1BLOB_USER_PRIORS[0,1]:.2f}] log(Jy/sr)", f"[{GAUSS1BLOB_PRIOR_RANGES[7,0]:.2f}, {GAUSS1BLOB_PRIOR_RANGES[7,1]:.2f}] log(Jy)"),
        ("Blob 1 Width (sigma)", f"[{GAUSS1BLOB_USER_PRIORS[1,0]:.2f}, {GAUSS1BLOB_USER_PRIORS[1,1]:.2f}] arcsec", f"[{GAUSS1BLOB_PRIOR_RANGES[8,0]:.3f}, {GAUSS1BLOB_PRIOR_RANGES[8,1]:.3f}] log(arcsec)"),
        ("Blob 1 Radial Distance", f"[{GAUSS1BLOB_USER_PRIORS[2,0]:.1f}, {GAUSS1BLOB_USER_PRIORS[2,1]:.1f}] arcsec", f"[{GAUSS1BLOB_PRIOR_RANGES[9,0]:.1f}, {GAUSS1BLOB_PRIOR_RANGES[9,1]:.1f}] arcsec (area)"),
        ("Blob 1 Azimuthal Angle", f"[{GAUSS1BLOB_USER_PRIORS[3,0]:.1f}, {GAUSS1BLOB_USER_PRIORS[3,1]:.1f}] deg", f"[{GAUSS1BLOB_PRIOR_RANGES[10,0]:.1f}, {GAUSS1BLOB_PRIOR_RANGES[10,1]:.1f}] deg"),
    ]
    for name, user_val, dyn_val in b1_labels:
        print(f"  {name:<30} {user_val:<22} {dyn_val:<24}")

    print("=" * 80)
    print("Compilation successful! Dynesty receives compiled (log_flux, log_sigma) bounds.")
