r"""
Prior Transformations for Dynesty Nested Sampling (Streamlined Edition)
========================================================================
In Bayesian nested sampling (Dynesty), sampling is performed on an N-dimensional
unit hypercube where coordinates $u_i \in [0, 1]$ represent cumulative probability
values. A prior transform function $\theta = F^{-1}(u)$ maps these coordinates
to physical parameter values.

Key Architecture & User Ergonomics:
-----------------------------------
1. Prior Cards:
   The prior ranges for a run live in a "prior card", a small text file in
   dyn_version/prior_cards/ named after the run (<session_name>.pc). A card names
   its model, then lists one [low, high] row per model parameter, in parameter order:

       model: twod_gauss1blob
       [6.417, 9.417], # 0: Ring Peak brightness [log10(Jy/sr)]
       [0.05, 4.0],    # 1: Ring Gaussian width sigma [arcsec]
       ...
       [330.0, 390.0]  # 10: Blob 1 Azimuthal Angle [deg]

   Everything after '#' is a comment, blank lines are ignored, and the commas
   between rows are optional. The <model_name>_legacy.pc cards hold the priors
   that used to be hardcoded in this file, and make good starting points.

2. Intuitive Units for Easy Editing:
   Cards are written in familiar physical units:
     - Surface Brightness:  log10(Jy/sr)
     - Gaussian Widths:     arcseconds (strictly positive, e.g. >= 0.05")
     - Spatial Positions:   arcseconds (ring radius, centroid offsets, clump separations)
     - Angular Coordinates: degrees (disk inclination, position angles, azimuthal angles)

3. Pre-Sampling Compilation (Zero Runtime Overhead):
   Interferometric visibilities directly constrain total integrated flux F = V(0).
   Sampling in (log10 Flux, log10 Sigma) eliminates the curved flux-size "funnel"
   degeneracy that plagues (peak, sigma) parameterizations.

   To avoid computing unit conversions repeatedly during sampling, the card's
   ranges are converted ONCE, when the card is loaded, into compiled Dynesty ranges:
     - Peak [log10(Jy/sr)] -> LogFlux [log10(Jy)]
     - Sigma [arcsec]      -> LogSigma [log10(arcsec)]
   Dynesty calls `ptform(u)` millions of times; during sampling, only ultra-fast
   linear interpolation and uniform-area disk transforms are executed. Zero conversion
   overhead occurs during the nested sampling run!

4. One Generic Transform Builder:
   `make_ptform(fittype, card_path)` builds the prior transform from the parameter
   layout in `model_defs.py` and the ranges in the card. To add a model: add it to
   `model_defs.py` and write a card for it.

Inspect a card (card ranges next to compiled Dynesty ranges) with:
    python prior_tform.py <card.pc> [fittype]
"""

import re
import functools
import numpy as np
from model_defs import get_spec, N_RING_PARAMS, N_BLOB_PARAMS

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
# PRIOR CARDS (Intuitive Units: log10(Jy/sr) and arcsec)
# =============================================================================

# One "[low, high]" pair
_CARD_ROW = re.compile(r"\[\s*([^\[\],]+?)\s*,\s*([^\[\],]+?)\s*\]")


def read_prior_card(path):
    """
    Reads a prior card.

    Returns
    -------
    model : str
        Model name from the card's 'model:' line.
    user_priors : numpy.ndarray
        (N, 2) array of [low, high] rows in intuitive units, in card order.
    """
    model = None
    rows = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            text = line.split('#', 1)[0].strip()
            if not text:
                continue
            if model is None:
                key, sep, value = text.partition(':')
                if not sep or key.strip().lower() != 'model' or not value.strip():
                    raise ValueError(f"{path} line {lineno}: the first entry must be 'model: <fittype>', got '{text}'")
                model = value.strip()
                continue
            pairs = _CARD_ROW.findall(text)
            leftover = _CARD_ROW.sub('', text).strip(' ,')
            if len(pairs) != 1 or leftover:
                raise ValueError(f"{path} line {lineno}: expected one '[low, high]' pair, got '{text}'")
            try:
                rows.append([float(pairs[0][0]), float(pairs[0][1])])
            except ValueError:
                raise ValueError(f"{path} line {lineno}: could not read numbers from '{text}'")
    if model is None:
        raise ValueError(f"{path}: no 'model: <fittype>' line found")
    return model, np.array(rows, dtype=float).reshape(-1, 2)


# =============================================================================
# COMPILED DYNESTY PRIOR RANGES (Pre-computed Ahead of Sampling)
# =============================================================================
# Card ranges are in intuitive units. The functions below convert them ONCE,
# when the card is loaded, into the (log_flux, log_sigma) ranges that Dynesty samples.

def compile_ring_block(ring_priors):
    """
    Compiles the 7 ring rows: ring peak [log10(Jy/sr)] -> LogFlux [log10(Jy)],
    ring sigma [arcsec] -> LogSigma [log10(arcsec)]. Other rows are unchanged.
    """
    comp = np.copy(ring_priors)
    comp[0, 0] = ring_peak_to_logflux(ring_priors[0, 0], rad_bounds=ring_priors[2], sigma_bounds=ring_priors[1])
    comp[0, 1] = ring_peak_to_logflux(ring_priors[0, 1], rad_bounds=ring_priors[2], sigma_bounds=ring_priors[1])
    comp[1, 0] = sigma_to_logsigma(ring_priors[1, 0])
    comp[1, 1] = sigma_to_logsigma(ring_priors[1, 1])
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


def compile_simgauss_block(simgauss_priors):
    """
    Compiles the simulated single-blob test priors. (Uses its own flux conversion with
    the lower/upper sigma bound, rounded to 2 decimals.)
    """
    comp = np.copy(simgauss_priors)
    comp[0, 0] = float(np.round(simgauss_priors[0, 0] + 2.0 * np.log10(simgauss_priors[1, 0] * ARCSEC_TO_RAD) + np.log10(2.0 * np.pi), 2))
    comp[0, 1] = float(np.round(simgauss_priors[0, 1] + 2.0 * np.log10(simgauss_priors[1, 1] * ARCSEC_TO_RAD) + np.log10(2.0 * np.pi), 2))
    comp[1, 0] = sigma_to_logsigma(simgauss_priors[1, 0])
    comp[1, 1] = sigma_to_logsigma(simgauss_priors[1, 1])
    return comp


def compile_priors(user_priors, spec):
    """
    Compiles a full card table for the model `spec`.
    Layout: [ring block][Prof. Offset if non-uniform][blob 1][blob 2]... (see model_defs.py).
    The Prof. Offset row is unchanged.
    """
    if spec.geometry == 'free':
        return compile_simgauss_block(user_priors)
    comp = np.copy(user_priors)
    if spec.has_ring:
        comp[:N_RING_PARAMS] = compile_ring_block(user_priors[:N_RING_PARAMS])
    b0 = spec.blob_start
    b1 = b0 + N_BLOB_PARAMS * spec.n_blobs
    if spec.n_blobs > 0:
        comp[b0:b1] = compile_blob_block(user_priors[b0:b1])
    return comp


def load_priors(card_path, fittype):
    """
    Reads and checks the prior card for `fittype`, then compiles it.

    Returns
    -------
    user_priors : numpy.ndarray
        (ndim, 2) card ranges in intuitive units.
    compiled_ranges : numpy.ndarray
        (ndim, 2) ranges sampled by Dynesty.
    """
    spec = get_spec(fittype)
    card_model, user_priors = read_prior_card(card_path)
    if get_spec(card_model).name != spec.name:
        raise ValueError(f"Prior card '{card_path}' is for model '{card_model}', but the fittype is '{fittype}'.")
    if len(user_priors) != spec.ndim:
        raise ValueError(f"Prior card '{card_path}' has {len(user_priors)} rows but '{spec.name}' has {spec.ndim} parameters.")
    bad = np.where(user_priors[:, 0] >= user_priors[:, 1])[0]
    if len(bad) > 0:
        raise ValueError(f"Prior card '{card_path}': rows {bad.tolist()} need low < high.")
    return user_priors, compile_priors(user_priors, spec)


# =============================================================================
# PRIOR TRANSFORM FUNCTIONS (Fast Sampling Execution: No Runtime Conversions)
# =============================================================================
# Every parameter is mapped linearly from the unit cube onto its [low, high]
# range, except blob radial distances, which use the uniform-area transform
# (so blobs are not over-sampled near the ring centre).
#
# Blob angular windows (set in the prior card) should be distinct and
# non-overlapping to break permutation symmetry between blobs and avoid
# label switching.

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


def make_ptform(fittype, card_path):
    """
    Builds the prior transform u -> theta for a fittype.

    The parameter layout comes from model_defs.py; the ranges from the prior card.
    Returns `_transform` with this model's ranges filled in (functools.partial).
    """
    spec = get_spec(fittype)
    _, ranges = load_priors(card_path, fittype)

    low = ranges[:, 0].copy()
    high = ranges[:, 1].copy()
    span = high - low
    area_dims = tuple(spec.area_dims)

    return functools.partial(_transform, low=low, high=high, span=span, area_dims=area_dims)


# =============================================================================
# CLI INSPECTION / SUMMARY DISPLAY
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) not in (2, 3):
        print("Usage: python prior_tform.py <card.pc> [fittype]")
        sys.exit(1)

    card_path = sys.argv[1]
    fittype = sys.argv[2] if len(sys.argv) == 3 else read_prior_card(card_path)[0]
    spec = get_spec(fittype)
    user_priors, compiled = load_priors(card_path, fittype)
    labels, units = spec.labels_units()

    print("=" * 96)
    print(f"PRIOR CARD: {card_path}   MODEL: {spec.name}   ({spec.ndim} parameters)")
    print("=" * 96)
    print(f"{'#':>3}  {'Parameter':<18} {'Card Prior':<24} {'Compiled Dynesty Prior':<26} {'Unit':<12}")
    print("-" * 96)
    for i, (label, unit) in enumerate(zip(labels, units)):
        card = f"[{user_priors[i, 0]:.3f}, {user_priors[i, 1]:.3f}]"
        comp = f"[{compiled[i, 0]:.3f}, {compiled[i, 1]:.3f}]"
        note = " (area)" if i in spec.area_dims else ""
        print(f"{i:>3}  {label:<18} {card:<24} {comp:<26} {unit}{note}")
    print("=" * 96)
