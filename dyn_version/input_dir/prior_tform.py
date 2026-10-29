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
   Ring models inherit identical ring prior boundaries from `COMMON_RING_PRIORS`.
   Modifying `COMMON_RING_PRIORS` automatically propagates to those model profiles,
   preserving Bayesian evidence comparability and Bayes factors across models.
   Ring-free models (`simgauss`, `resid_gauss1blob`, `resid_gauss3blob`) do not use `COMMON_RING_PRIORS`.

4. Familiar Table Formatting:
   The user configuration section is styled as a clean, readable table.

5. One Generic Transform Builder:
   `make_ptform(fittype)` builds every `<fittype>_ptform` function from the parameter
   layout in `model_defs.py` and the ranges in this file. There are no hand-indexed
   per-model transforms to keep in sync. To add a model: add it to `model_defs.py`,
   add its prior table here, and register it in `PRIOR_RANGES`.
"""

import functools
import numpy as np
from model_defs import get_spec

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

# twod_gaussring_nonuni: Non-Uniform Gaussian Ring (parameter 7)
# Azimuthal offset angle of the emission peak on the disk [deg] (180 deg periodicity for m=2 mode)
GAUSSRING_NONUNI_USER_PRIORS = np.array([
    [0.0, 180.0]    # 7: Profile Azimuthal Offset [deg] (East of North in disk frame)
], dtype=float)

# twod_gauss1blob: Single Gaussian Blob (parameters 7-10)
GAUSS1BLOB_USER_PRIORS = np.array([
    [6.030, 8.530], # 7:  Blob 1 Peak [log10(Jy/sr)]   -> converted to LogFlux [-5.5, -3.0]
    [0.05, 0.4],    # 8:  Blob 1 Width sigma [arcsec]  -> converted to LogSigma [-1.301, -0.398]
    [4.0, 10.0],    # 9:  Blob 1 Radial Dist [arcsec]  (sampled uniformly in area on disk)
    [330.0, 390.0]  # 10: Blob 1 Azimuthal Angle [deg] (East of North, counter-clockwise)
], dtype=float)

# twod_gauss2blob: Two Distinct Clumps (parameters 7-14)
GAUSS2BLOB_USER_PRIORS = np.array([
    # Blob 1: North clump (YMC 6 complex)
    [6.030, 8.530], # 7:  Blob 1 Peak [log10(Jy/sr)]   -> converted to LogFlux [-5.5, -3.0]
    [0.05, 0.4],    # 8:  Blob 1 Width sigma [arcsec]  -> converted to LogSigma [-1.301, -0.398]
    [4.0, 10.0],    # 9:  Blob 1 Radial Dist [arcsec]  (sampled uniformly in area on disk)
    [330.0, 390.0], # 10: Blob 1 Angle [deg] (East of North, CCW)
    # Blob 2: South-West clump (YMC 17/18 complex)
    [6.030, 8.530], # 11: Blob 2 Peak [log10(Jy/sr)]   -> converted to LogFlux [-5.5, -3.0]
    [0.05, 0.4],    # 12: Blob 2 Width sigma [arcsec]  -> converted to LogSigma [-1.301, -0.398]
    [4.0, 10.0],    # 13: Blob 2 Radial Dist [arcsec]  (sampled uniformly in area on disk)
    [177.0, 205.0]  # 14: Blob 2 Angle [deg] (East of North, CCW)
], dtype=float)

# twod_gauss3blob: Three Distinct Clumps (parameters 7-18)
GAUSS3BLOB_USER_PRIORS = np.array([
    # Blob 1: North clump (YMC 6 complex, disk angle ~348 deg)
    [6.030, 8.530], # 7:  Blob 1 Peak [log10(Jy/sr)]   -> converted to LogFlux [-5.5, -3.0]
    [0.05, 0.4],    # 8:  Blob 1 Width sigma [arcsec]  -> converted to LogSigma [-1.301, -0.398]
    [4.0, 10.0],    # 9:  Blob 1 Radial Dist [arcsec]  (sampled uniformly in area on disk)
    [330.0, 390.0], # 10: Blob 1 Angle [deg] (East of North, CCW)
    # Blob 2: South-West clump (YMC 17/18 complex, disk angle ~182-185 deg)
    [6.030, 8.530], # 11: Blob 2 Peak [log10(Jy/sr)]   -> converted to LogFlux [-5.5, -3.0]
    [0.05, 0.4],    # 12: Blob 2 Width sigma [arcsec]  -> converted to LogSigma [-1.301, -0.398]
    [4.0, 10.0],    # 13: Blob 2 Radial Dist [arcsec]  (sampled uniformly in area on disk)
    [177.0, 205.0], # 14: Blob 2 Angle [deg] (East of North, CCW)
    # Blob 3: South-East clump (YMC 15 complex, disk angle ~171 deg)
    [6.030, 8.530], # 15: Blob 3 Peak [log10(Jy/sr)]   -> converted to LogFlux [-5.5, -3.0]
    [0.05, 0.4],    # 16: Blob 3 Width sigma [arcsec]  -> converted to LogSigma [-1.301, -0.398]
    [4.0, 10.0],    # 17: Blob 3 Radial Dist [arcsec]  (sampled uniformly in area on disk)
    [155.0, 177.0]  # 18: Blob 3 Angle [deg] (East of North, CCW)
], dtype=float)

# resid_gauss1blob: One Clump on a residual map (4 parameters).
# Disk PA, dRA, and dDec are fixed inside the model and are not sampled.
RESID1BLOB_USER_PRIORS = np.array([
    # Blob 1:
    [6.030, 8.530], # 0:  Blob 1 Peak [log10(Jy/sr)]   -> converted to LogFlux [-5.5, -3.0]
    [0.05, 0.4],    # 1:  Blob 1 Width sigma [arcsec]  -> converted to LogSigma [-1.301, -0.398]
    [7.5, 8.5],    # 2:  Blob 1 Radial Dist [arcsec]  (sampled uniformly in area on disk)
    [167.5, 175.0]  # 3:  Blob 1 Angle [deg] (East of North, CCW)
], dtype=float)

# resid_gauss3blob: Three Distinct Clumps on a residual map (12 parameters).
# Disk PA, dRA, and dDec are fixed inside the model and are not sampled.
RESID3BLOB_USER_PRIORS = np.array([
    # Blob 1:
    [6.030, 8.530], # 0:  Blob 1 Peak [log10(Jy/sr)]   -> converted to LogFlux [-5.5, -3.0]
    [0.05, 0.4],    # 1:  Blob 1 Width sigma [arcsec]  -> converted to LogSigma [-1.301, -0.398]
    [2.5, 5.0],    # 2:  Blob 1 Radial Dist [arcsec]  (sampled uniformly in area on disk)
    [230.0, 310.0], # 3:  Blob 1 Angle [deg] (East of North, CCW)
    # Blob 2:
    [6.030, 8.530], # 4:  Blob 2 Peak [log10(Jy/sr)]   -> converted to LogFlux [-5.5, -3.0]
    [0.05, 0.4],    # 5:  Blob 2 Width sigma [arcsec]  -> converted to LogSigma [-1.301, -0.398]
    [2.5, 5.0],    # 6:  Blob 2 Radial Dist [arcsec]  (sampled uniformly in area on disk)
    [230.0, 310.0], # 7:  Blob 2 Angle [deg] (East of North, CCW)
    # Blob 3:
    [6.030, 8.530], # 8:  Blob 3 Peak [log10(Jy/sr)]   -> converted to LogFlux [-5.5, -3.0]
    [0.05, 0.4],    # 9:  Blob 3 Width sigma [arcsec]  -> converted to LogSigma [-1.301, -0.398]
    [2.5, 5.0],    # 10: Blob 3 Radial Dist [arcsec]  (sampled uniformly in area on disk)
    [230.0, 310.0]  # 11: Blob 3 Angle [deg] (East of North, CCW)
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
# The tables above are in intuitive units. The functions below convert them ONCE,
# at import time, into the (log_flux, log_sigma) ranges that Dynesty samples.

def compile_ring_block():
    """
    Compiles the shared ring priors: ring peak [log10(Jy/sr)] -> LogFlux [log10(Jy)],
    ring sigma [arcsec] -> LogSigma [log10(arcsec)]. Other rows are unchanged.
    """
    comp = np.copy(COMMON_RING_PRIORS)
    comp[0, 0] = ring_peak_to_logflux(COMMON_RING_PRIORS[0, 0], rad_bounds=COMMON_RING_PRIORS[2], sigma_bounds=COMMON_RING_PRIORS[1])
    comp[0, 1] = ring_peak_to_logflux(COMMON_RING_PRIORS[0, 1], rad_bounds=COMMON_RING_PRIORS[2], sigma_bounds=COMMON_RING_PRIORS[1])
    comp[1, 0] = sigma_to_logsigma(COMMON_RING_PRIORS[1, 0])
    comp[1, 1] = sigma_to_logsigma(COMMON_RING_PRIORS[1, 1])
    return comp


def compile_blob_block(user_priors):
    """
    Compiles any stack of blob priors (4 rows per blob: Peak, Sigma, Dist, Angle):
    blob peak [log10(Jy/sr)] -> LogFlux [log10(Jy)], blob sigma [arcsec] -> LogSigma [log10(arcsec)].
    Dist and Angle rows are unchanged.
    """
    comp = np.copy(user_priors)
    for k in range(0, len(user_priors), 4):
        comp[k, 0] = blob_peak_to_logflux(user_priors[k, 0], sigma_bounds=user_priors[k + 1])
        comp[k, 1] = blob_peak_to_logflux(user_priors[k, 1], sigma_bounds=user_priors[k + 1])
        comp[k + 1, 0] = sigma_to_logsigma(user_priors[k + 1, 0])
        comp[k + 1, 1] = sigma_to_logsigma(user_priors[k + 1, 1])
    return comp


def compile_simgauss_block():
    """
    Compiles the simulated single-blob test priors. (Uses its own flux conversion with
    the lower/upper sigma bound, rounded to 2 decimals.)
    """
    comp = np.copy(SIMGAUSS_USER_PRIORS)
    comp[0, 0] = float(np.round(SIMGAUSS_USER_PRIORS[0, 0] + 2.0 * np.log10(SIMGAUSS_USER_PRIORS[1, 0] * ARCSEC_TO_RAD) + np.log10(2.0 * np.pi), 2))
    comp[0, 1] = float(np.round(SIMGAUSS_USER_PRIORS[0, 1] + 2.0 * np.log10(SIMGAUSS_USER_PRIORS[1, 1] * ARCSEC_TO_RAD) + np.log10(2.0 * np.pi), 2))
    comp[1, 0] = sigma_to_logsigma(SIMGAUSS_USER_PRIORS[1, 0])
    comp[1, 1] = sigma_to_logsigma(SIMGAUSS_USER_PRIORS[1, 1])
    return comp


# Compile ranges immediately upon module import.
# Parameter order for ring models is [ring block][Prof. Offset if non-uniform][blob 1][blob 2]...
# (the layout itself is defined in model_defs.py).
RING_PRIOR_RANGES = compile_ring_block()
NONUNI_RING_PRIOR_RANGES = np.vstack([RING_PRIOR_RANGES, GAUSSRING_NONUNI_USER_PRIORS])
GAUSS1BLOB_PRIOR_RANGES = np.vstack([RING_PRIOR_RANGES, compile_blob_block(GAUSS1BLOB_USER_PRIORS)])
GAUSS2BLOB_PRIOR_RANGES = np.vstack([RING_PRIOR_RANGES, compile_blob_block(GAUSS2BLOB_USER_PRIORS)])
GAUSS3BLOB_PRIOR_RANGES = np.vstack([RING_PRIOR_RANGES, compile_blob_block(GAUSS3BLOB_USER_PRIORS)])
GAUSS3BLOB_NONUNI_PRIOR_RANGES = np.vstack([NONUNI_RING_PRIOR_RANGES, compile_blob_block(GAUSS3BLOB_USER_PRIORS)])
RESID1BLOB_PRIOR_RANGES = compile_blob_block(RESID1BLOB_USER_PRIORS)
RESID3BLOB_PRIOR_RANGES = compile_blob_block(RESID3BLOB_USER_PRIORS)
SIMGAUSS_PRIOR_RANGES = compile_simgauss_block()

# Compiled ranges, and the raw (intuitive-unit) blob priors, looked up by fittype name.
# Used by the transforms below and by the summary table in visualize_dynesty.py.
PRIOR_RANGES = {
    'twod_gaussring': RING_PRIOR_RANGES,
    'twod_gaussring_nonuni': NONUNI_RING_PRIOR_RANGES,
    'twod_gauss1blob': GAUSS1BLOB_PRIOR_RANGES,
    'twod_gauss2blob': GAUSS2BLOB_PRIOR_RANGES,
    'twod_gauss3blob': GAUSS3BLOB_PRIOR_RANGES,
    'twod_gauss3blob_nonuni': GAUSS3BLOB_NONUNI_PRIOR_RANGES,
    'resid_gauss1blob': RESID1BLOB_PRIOR_RANGES,
    'resid_gauss3blob': RESID3BLOB_PRIOR_RANGES,
    'simgauss': SIMGAUSS_PRIOR_RANGES,
}

BLOB_USER_PRIORS = {
    'twod_gauss1blob': GAUSS1BLOB_USER_PRIORS,
    'twod_gauss2blob': GAUSS2BLOB_USER_PRIORS,
    'twod_gauss3blob': GAUSS3BLOB_USER_PRIORS,
    'twod_gauss3blob_nonuni': GAUSS3BLOB_USER_PRIORS,
    'resid_gauss1blob': RESID1BLOB_USER_PRIORS,
    'resid_gauss3blob': RESID3BLOB_USER_PRIORS,
    'simgauss': SIMGAUSS_USER_PRIORS,
}


# =============================================================================
# PRIOR TRANSFORM FUNCTIONS (Fast Sampling Execution: No Runtime Conversions)
# =============================================================================
# Every parameter is mapped linearly from the unit cube onto its [low, high]
# range, except blob radial distances, which use the uniform-area transform
# (so blobs are not over-sampled near the ring centre).
#
# Blob angular windows (set in the USER_PRIORS tables above) are deliberately
# distinct and non-overlapping (North, South-West, South-East clumps) to break
# permutation symmetry between blobs and avoid label switching.

def _transform(u, low, high, span, area_dims):
    """
    The prior transform itself: unit hypercube u -> physical parameters.

    Linear in u between low and high, except the blob distances (area_dims),
    which are sampled uniformly in area. This is a top-level function (not nested
    inside make_ptform) so that multiprocessing can pickle it and send it to the
    worker processes.
    """
    u = np.asarray(u)
    v = low + u * span
    for i in area_dims:
        v[i] = uniform_area_radius(u[i], low[i], high[i])
    return v


def make_ptform(fittype):
    """
    Builds the prior transform u -> theta for a fittype.

    The parameter layout comes from model_defs.py; the ranges from PRIOR_RANGES.
    Returns `_transform` with this model's ranges filled in (functools.partial).
    """
    spec = get_spec(fittype)
    ranges = PRIOR_RANGES[spec.name]
    if len(ranges) != spec.ndim:
        raise ValueError(f"Prior table for '{spec.name}' has {len(ranges)} rows but the model has {spec.ndim} parameters.")

    low = ranges[:, 0].copy()
    high = ranges[:, 1].copy()
    span = high - low
    area_dims = tuple(spec.area_dims)

    return functools.partial(_transform, low=low, high=high, span=span, area_dims=area_dims)


twod_gaussring_ptform = make_ptform('twod_gaussring')                    # 7 parameters
twod_gaussring_nonuni_ptform = make_ptform('twod_gaussring_nonuni')      # 8 parameters
twod_gauss1blob_ptform = make_ptform('twod_gauss1blob')                  # 11 parameters
twod_gauss2blob_ptform = make_ptform('twod_gauss2blob')                  # 15 parameters
twod_gauss3blob_ptform = make_ptform('twod_gauss3blob')                  # 19 parameters
twod_gauss3blob_nonuni_ptform = make_ptform('twod_gauss3blob_nonuni')    # 20 parameters
resid_gauss1blob_ptform = make_ptform('resid_gauss1blob')                # 4 parameters (PA, dRA, dDec fixed)
resid_gauss3blob_ptform = make_ptform('resid_gauss3blob')                # 12 parameters (PA, dRA, dDec fixed)
simgauss_ptform = make_ptform('simgauss')                                # 7 parameters

# Alternate fittype spellings (see ALIASES in model_defs.py)
twod_gaussring_nonuni_3blob_ptform = twod_gauss3blob_nonuni_ptform
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

    print("\nNon-Uniform Ring Additional Parameter (twod_gaussring_nonuni / twod_gauss3blob_nonuni):")
    nonuni_labels = [
        ("Profile Azimuthal Offset", f"[{GAUSSRING_NONUNI_USER_PRIORS[0,0]:.1f}, {GAUSSRING_NONUNI_USER_PRIORS[0,1]:.1f}] deg", f"[{NONUNI_RING_PRIOR_RANGES[7,0]:.1f}, {NONUNI_RING_PRIOR_RANGES[7,1]:.1f}] deg"),
    ]
    for name, user_val, dyn_val in nonuni_labels:
        print(f"  {name:<30} {user_val:<22} {dyn_val:<24}")

    print("\nBlob 1 Parameters (twod_gauss1blob / twod_gauss2blob / twod_gauss3blob / twod_gauss3blob_nonuni):")
    b1_labels = [
        ("Blob 1 Surface Brightness", f"[{GAUSS1BLOB_USER_PRIORS[0,0]:.2f}, {GAUSS1BLOB_USER_PRIORS[0,1]:.2f}] log(Jy/sr)", f"[{GAUSS1BLOB_PRIOR_RANGES[7,0]:.2f}, {GAUSS1BLOB_PRIOR_RANGES[7,1]:.2f}] log(Jy)"),
        ("Blob 1 Width (sigma)", f"[{GAUSS1BLOB_USER_PRIORS[1,0]:.2f}, {GAUSS1BLOB_USER_PRIORS[1,1]:.2f}] arcsec", f"[{GAUSS1BLOB_PRIOR_RANGES[8,0]:.3f}, {GAUSS1BLOB_PRIOR_RANGES[8,1]:.3f}] log(arcsec)"),
        ("Blob 1 Radial Distance", f"[{GAUSS1BLOB_USER_PRIORS[2,0]:.1f}, {GAUSS1BLOB_USER_PRIORS[2,1]:.1f}] arcsec", f"[{GAUSS1BLOB_PRIOR_RANGES[9,0]:.1f}, {GAUSS1BLOB_PRIOR_RANGES[9,1]:.1f}] arcsec (area)"),
        ("Blob 1 Azimuthal Angle (North)", f"[{GAUSS1BLOB_USER_PRIORS[3,0]:.1f}, {GAUSS1BLOB_USER_PRIORS[3,1]:.1f}] deg", f"[{GAUSS1BLOB_PRIOR_RANGES[10,0]:.1f}, {GAUSS1BLOB_PRIOR_RANGES[10,1]:.1f}] deg"),
    ]
    for name, user_val, dyn_val in b1_labels:
        print(f"  {name:<30} {user_val:<22} {dyn_val:<24}")

    print("\nBlob 2 Parameters (twod_gauss2blob / twod_gauss3blob / twod_gauss3blob_nonuni):")
    b2_labels = [
        ("Blob 2 Surface Brightness", f"[{GAUSS2BLOB_USER_PRIORS[4,0]:.2f}, {GAUSS2BLOB_USER_PRIORS[4,1]:.2f}] log(Jy/sr)", f"[{GAUSS2BLOB_PRIOR_RANGES[11,0]:.2f}, {GAUSS2BLOB_PRIOR_RANGES[11,1]:.2f}] log(Jy)"),
        ("Blob 2 Width (sigma)", f"[{GAUSS2BLOB_USER_PRIORS[5,0]:.2f}, {GAUSS2BLOB_USER_PRIORS[5,1]:.2f}] arcsec", f"[{GAUSS2BLOB_PRIOR_RANGES[12,0]:.3f}, {GAUSS2BLOB_PRIOR_RANGES[12,1]:.3f}] log(arcsec)"),
        ("Blob 2 Radial Distance", f"[{GAUSS2BLOB_USER_PRIORS[6,0]:.1f}, {GAUSS2BLOB_USER_PRIORS[6,1]:.1f}] arcsec", f"[{GAUSS2BLOB_PRIOR_RANGES[13,0]:.1f}, {GAUSS2BLOB_PRIOR_RANGES[13,1]:.1f}] arcsec (area)"),
        ("Blob 2 Azimuthal Angle (SW)", f"[{GAUSS2BLOB_USER_PRIORS[7,0]:.1f}, {GAUSS2BLOB_USER_PRIORS[7,1]:.1f}] deg", f"[{GAUSS2BLOB_PRIOR_RANGES[14,0]:.1f}, {GAUSS2BLOB_PRIOR_RANGES[14,1]:.1f}] deg"),
    ]
    for name, user_val, dyn_val in b2_labels:
        print(f"  {name:<30} {user_val:<22} {dyn_val:<24}")

    print("\nBlob 3 Parameters (twod_gauss3blob / twod_gauss3blob_nonuni):")
    b3_labels = [
        ("Blob 3 Surface Brightness", f"[{GAUSS3BLOB_USER_PRIORS[8,0]:.2f}, {GAUSS3BLOB_USER_PRIORS[8,1]:.2f}] log(Jy/sr)", f"[{GAUSS3BLOB_PRIOR_RANGES[15,0]:.2f}, {GAUSS3BLOB_PRIOR_RANGES[15,1]:.2f}] log(Jy)"),
        ("Blob 3 Width (sigma)", f"[{GAUSS3BLOB_USER_PRIORS[9,0]:.2f}, {GAUSS3BLOB_USER_PRIORS[9,1]:.2f}] arcsec", f"[{GAUSS3BLOB_PRIOR_RANGES[16,0]:.3f}, {GAUSS3BLOB_PRIOR_RANGES[16,1]:.3f}] log(arcsec)"),
        ("Blob 3 Radial Distance", f"[{GAUSS3BLOB_USER_PRIORS[10,0]:.1f}, {GAUSS3BLOB_USER_PRIORS[10,1]:.1f}] arcsec", f"[{GAUSS3BLOB_PRIOR_RANGES[17,0]:.1f}, {GAUSS3BLOB_PRIOR_RANGES[17,1]:.1f}] arcsec (area)"),
        ("Blob 3 Azimuthal Angle (SE)", f"[{GAUSS3BLOB_USER_PRIORS[11,0]:.1f}, {GAUSS3BLOB_USER_PRIORS[11,1]:.1f}] deg", f"[{GAUSS3BLOB_PRIOR_RANGES[18,0]:.1f}, {GAUSS3BLOB_PRIOR_RANGES[18,1]:.1f}] deg"),
    ]
    for name, user_val, dyn_val in b3_labels:
        print(f"  {name:<30} {user_val:<22} {dyn_val:<24}")

    print("\nResidual 1-Blob Parameters (resid_gauss1blob; PA, dRA, dDec fixed in the model):")
    resid1_labels = [
        ("Blob 1 Surface Brightness", f"[{RESID1BLOB_USER_PRIORS[0,0]:.2f}, {RESID1BLOB_USER_PRIORS[0,1]:.2f}] log(Jy/sr)", f"[{RESID1BLOB_PRIOR_RANGES[0,0]:.2f}, {RESID1BLOB_PRIOR_RANGES[0,1]:.2f}] log(Jy)"),
        ("Blob 1 Width (sigma)", f"[{RESID1BLOB_USER_PRIORS[1,0]:.2f}, {RESID1BLOB_USER_PRIORS[1,1]:.2f}] arcsec", f"[{RESID1BLOB_PRIOR_RANGES[1,0]:.3f}, {RESID1BLOB_PRIOR_RANGES[1,1]:.3f}] log(arcsec)"),
        ("Blob 1 Radial Distance", f"[{RESID1BLOB_USER_PRIORS[2,0]:.1f}, {RESID1BLOB_USER_PRIORS[2,1]:.1f}] arcsec", f"[{RESID1BLOB_PRIOR_RANGES[2,0]:.1f}, {RESID1BLOB_PRIOR_RANGES[2,1]:.1f}] arcsec (area)"),
        ("Blob 1 Azimuthal Angle", f"[{RESID1BLOB_USER_PRIORS[3,0]:.1f}, {RESID1BLOB_USER_PRIORS[3,1]:.1f}] deg", f"[{RESID1BLOB_PRIOR_RANGES[3,0]:.1f}, {RESID1BLOB_PRIOR_RANGES[3,1]:.1f}] deg"),
    ]
    for name, user_val, dyn_val in resid1_labels:
        print(f"  {name:<30} {user_val:<22} {dyn_val:<24}")

    print("\nResidual Blob Parameters (resid_gauss3blob; PA, dRA, dDec fixed in the model):")
    resid_labels = [
        ("Blob 1 Surface Brightness", f"[{RESID3BLOB_USER_PRIORS[0,0]:.2f}, {RESID3BLOB_USER_PRIORS[0,1]:.2f}] log(Jy/sr)", f"[{RESID3BLOB_PRIOR_RANGES[0,0]:.2f}, {RESID3BLOB_PRIOR_RANGES[0,1]:.2f}] log(Jy)"),
        ("Blob 1 Width (sigma)", f"[{RESID3BLOB_USER_PRIORS[1,0]:.2f}, {RESID3BLOB_USER_PRIORS[1,1]:.2f}] arcsec", f"[{RESID3BLOB_PRIOR_RANGES[1,0]:.3f}, {RESID3BLOB_PRIOR_RANGES[1,1]:.3f}] log(arcsec)"),
        ("Blob 1 Radial Distance", f"[{RESID3BLOB_USER_PRIORS[2,0]:.1f}, {RESID3BLOB_USER_PRIORS[2,1]:.1f}] arcsec", f"[{RESID3BLOB_PRIOR_RANGES[2,0]:.1f}, {RESID3BLOB_PRIOR_RANGES[2,1]:.1f}] arcsec (area)"),
        ("Blob 1 Azimuthal Angle", f"[{RESID3BLOB_USER_PRIORS[3,0]:.1f}, {RESID3BLOB_USER_PRIORS[3,1]:.1f}] deg", f"[{RESID3BLOB_PRIOR_RANGES[3,0]:.1f}, {RESID3BLOB_PRIOR_RANGES[3,1]:.1f}] deg"),
        ("Blob 2 Surface Brightness", f"[{RESID3BLOB_USER_PRIORS[4,0]:.2f}, {RESID3BLOB_USER_PRIORS[4,1]:.2f}] log(Jy/sr)", f"[{RESID3BLOB_PRIOR_RANGES[4,0]:.2f}, {RESID3BLOB_PRIOR_RANGES[4,1]:.2f}] log(Jy)"),
        ("Blob 2 Width (sigma)", f"[{RESID3BLOB_USER_PRIORS[5,0]:.2f}, {RESID3BLOB_USER_PRIORS[5,1]:.2f}] arcsec", f"[{RESID3BLOB_PRIOR_RANGES[5,0]:.3f}, {RESID3BLOB_PRIOR_RANGES[5,1]:.3f}] log(arcsec)"),
        ("Blob 2 Radial Distance", f"[{RESID3BLOB_USER_PRIORS[6,0]:.1f}, {RESID3BLOB_USER_PRIORS[6,1]:.1f}] arcsec", f"[{RESID3BLOB_PRIOR_RANGES[6,0]:.1f}, {RESID3BLOB_PRIOR_RANGES[6,1]:.1f}] arcsec (area)"),
        ("Blob 2 Azimuthal Angle", f"[{RESID3BLOB_USER_PRIORS[7,0]:.1f}, {RESID3BLOB_USER_PRIORS[7,1]:.1f}] deg", f"[{RESID3BLOB_PRIOR_RANGES[7,0]:.1f}, {RESID3BLOB_PRIOR_RANGES[7,1]:.1f}] deg"),
        ("Blob 3 Surface Brightness", f"[{RESID3BLOB_USER_PRIORS[8,0]:.2f}, {RESID3BLOB_USER_PRIORS[8,1]:.2f}] log(Jy/sr)", f"[{RESID3BLOB_PRIOR_RANGES[8,0]:.2f}, {RESID3BLOB_PRIOR_RANGES[8,1]:.2f}] log(Jy)"),
        ("Blob 3 Width (sigma)", f"[{RESID3BLOB_USER_PRIORS[9,0]:.2f}, {RESID3BLOB_USER_PRIORS[9,1]:.2f}] arcsec", f"[{RESID3BLOB_PRIOR_RANGES[9,0]:.3f}, {RESID3BLOB_PRIOR_RANGES[9,1]:.3f}] log(arcsec)"),
        ("Blob 3 Radial Distance", f"[{RESID3BLOB_USER_PRIORS[10,0]:.1f}, {RESID3BLOB_USER_PRIORS[10,1]:.1f}] arcsec", f"[{RESID3BLOB_PRIOR_RANGES[10,0]:.1f}, {RESID3BLOB_PRIOR_RANGES[10,1]:.1f}] arcsec (area)"),
        ("Blob 3 Azimuthal Angle", f"[{RESID3BLOB_USER_PRIORS[11,0]:.1f}, {RESID3BLOB_USER_PRIORS[11,1]:.1f}] deg", f"[{RESID3BLOB_PRIOR_RANGES[11,0]:.1f}, {RESID3BLOB_PRIOR_RANGES[11,1]:.1f}] deg"),
    ]
    for name, user_val, dyn_val in resid_labels:
        print(f"  {name:<30} {user_val:<22} {dyn_val:<24}")

    print("=" * 80)
    print("Compilation successful! Dynesty receives compiled (log_flux, log_sigma) bounds.")
