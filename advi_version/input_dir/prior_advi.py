r"""
Prior Distributions and Coordinate Transformations for ADVI
============================================================
In Automatic Differentiation Variational Inference (ADVI), bounded physical
parameters $\theta \in \text{supp}(p)$ are mapped bijectively to unconstrained real
coordinates $\zeta \in \mathbb{R}^D$ (Kucukelbir et al. 2017).

Key Principles:
1. Coordinate Transformation ($T: \text{supp}(p) \to \mathbb{R}^D$):
   For bounded interval $\theta_i \in [a_i, b_i]$:
       $\zeta_i = \text{logit}\left(\frac{\theta_i - a_i}{b_i - a_i}\right) = \ln\left(\frac{\theta_i - a_i}{b_i - \theta_i}\right)$
   Inverse mapping ($T^{-1}: \mathbb{R}^D \to \text{supp}(p)$):
       $u_i = \sigma(\zeta_i) = \frac{1}{1 + e^{-\zeta_i}}$
       $\theta_i = a_i + (b_i - a_i) \cdot \sigma(\zeta_i)$

2. Radial Distance on Disk (Uniform Area Prior):
   For blob radial separation $r \in [r_{\min}, r_{\max}]$ with non-singular disk area prior $p(r) \propto r$:
       $r = \sqrt{r_{\min}^2 + u \cdot (r_{\max}^2 - r_{\min}^2)}, \quad u = \sigma(\zeta)$

3. Exact Prior + Jacobian Density & Analytic Gradients:
   In unconstrained space, the joint prior density and transformation Jacobian is:
       $\ln p(\zeta) = \ln p(\theta(\zeta)) + \sum_{i=1}^D \ln \left| \frac{\partial \theta_i}{\partial \zeta_i} \right| = \sum_{i=1}^D [\ln \sigma(\zeta_i) + \ln(1 - \sigma(\zeta_i))]$
   The exact analytical gradient with respect to $\zeta$ is:
       $\nabla_\zeta \ln p(\zeta) = 1 - 2\sigma(\zeta)$
   Zero numerical approximations are required for the prior term!
"""

import numpy as np

# Physical conversion constant: radians per arcsecond
ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)

# =============================================================================
# Intuitive User Prior Ranges & Helper Converters
# =============================================================================

def sigma_to_logsigma(sigma_arcsec):
    sigma = float(sigma_arcsec)
    if sigma <= 0.0:
        raise ValueError(f"Gaussian sigma must be strictly positive (> 0 arcsec), got {sigma}")
    return float(np.round(np.log10(sigma), 3))

def blob_peak_to_logflux(peak_jysr, sigma_bounds=None, sigma_arcsec=None):
    if sigma_arcsec is not None:
        s_ref = float(sigma_arcsec)
    elif sigma_bounds is not None:
        s_ref = float(np.sqrt(sigma_bounds[0] * sigma_bounds[1]))
    else:
        s_ref = 0.22360679774997896

    sigma_rad = s_ref * ARCSEC_TO_RAD
    area_sr = 2.0 * np.pi * (sigma_rad**2)
    return float(np.round(peak_jysr + np.log10(area_sr), 3))

def ring_peak_to_logflux(peak_jysr, rad_bounds=None, sigma_bounds=None, inc_deg=65.0):
    r_ref = float(np.sqrt(rad_bounds[0] * rad_bounds[1])) if rad_bounds is not None else 5.477225575051661
    s_ref = float(np.sqrt(sigma_bounds[0] * sigma_bounds[1])) if sigma_bounds is not None else 0.4472135954999579
    inc_rad = np.radians(float(inc_deg))

    area_sr = ((2.0 * np.pi)**1.5) * (r_ref * ARCSEC_TO_RAD) * (s_ref * ARCSEC_TO_RAD) * np.cos(inc_rad)
    return float(np.round(peak_jysr + np.log10(area_sr), 3))


# User configurable prior ranges (in intuitive units)
COMMON_RING_PRIORS = np.array([
    [6.417, 9.417], # 0: Ring Peak brightness [log10(Jy/sr)] -> converted to LogFlux [-3.0, 0.0] Jy
    [0.05, 4.0],    # 1: Ring Gaussian width sigma [arcsec]  -> converted to LogSigma [-1.301, 0.602]
    [3.0, 10.0],    # 2: Ring center radius [arcsec]
    [0.0, 85.0],    # 3: Disk inclination angle [deg]
    [0.0, 180.0],   # 4: Disk position angle PA [deg] (East of North, 180 deg periodicity)
    [-4.0, 4.0],    # 5: Centroid RA offset dRA [arcsec]
    [-4.0, 4.0]     # 6: Centroid Dec offset dDec [arcsec]
], dtype=float)

GAUSS1BLOB_USER_PRIORS = np.array([
    [6.030, 8.530], # 7:  Blob 1 Peak [log10(Jy/sr)]   -> converted to LogFlux [-5.5, -3.0]
    [0.05, 0.4],    # 8:  Blob 1 Width sigma [arcsec]  -> converted to LogSigma [-1.301, -0.398]
    [4.0, 10.0],    # 9:  Blob 1 Radial Dist [arcsec]  (sampled uniformly in area on disk)
    [330.0, 390.0]  # 10: Blob 1 Azimuthal Angle [deg] (East of North, counter-clockwise)
], dtype=float)


def compile_advi_priors():
    """
    Compiles intuitive physical priors into parameter bounds [low, high] for ADVI.
    """
    # 1. Ring priors
    r_comp = np.copy(COMMON_RING_PRIORS)
    r_comp[0, 0] = ring_peak_to_logflux(COMMON_RING_PRIORS[0, 0], rad_bounds=COMMON_RING_PRIORS[2], sigma_bounds=COMMON_RING_PRIORS[1])
    r_comp[0, 1] = ring_peak_to_logflux(COMMON_RING_PRIORS[0, 1], rad_bounds=COMMON_RING_PRIORS[2], sigma_bounds=COMMON_RING_PRIORS[1])
    r_comp[1, 0] = sigma_to_logsigma(COMMON_RING_PRIORS[1, 0])
    r_comp[1, 1] = sigma_to_logsigma(COMMON_RING_PRIORS[1, 1])

    # 2. Blob 1 priors
    b1_comp = np.copy(GAUSS1BLOB_USER_PRIORS)
    b1_comp[0, 0] = blob_peak_to_logflux(GAUSS1BLOB_USER_PRIORS[0, 0], sigma_bounds=GAUSS1BLOB_USER_PRIORS[1])
    b1_comp[0, 1] = blob_peak_to_logflux(GAUSS1BLOB_USER_PRIORS[0, 1], sigma_bounds=GAUSS1BLOB_USER_PRIORS[1])
    b1_comp[1, 0] = sigma_to_logsigma(GAUSS1BLOB_USER_PRIORS[1, 0])
    b1_comp[1, 1] = sigma_to_logsigma(GAUSS1BLOB_USER_PRIORS[1, 1])

    gauss1blob_ranges = np.vstack([r_comp, b1_comp])
    return r_comp, gauss1blob_ranges

RING_PRIOR_RANGES, GAUSS1BLOB_PRIOR_RANGES = compile_advi_priors()


# =============================================================================
# Numerically Stable Math Primitives
# =============================================================================

def sigmoid(z):
    """
    Numerically stable logistic sigmoid function: sigma(z) = 1 / (1 + exp(-z)).
    """
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

def logit(u, eps=1e-12):
    """
    Numerically stable logit function: logit(u) = ln(u / (1 - u)).
    """
    u = np.clip(np.asarray(u, dtype=float), eps, 1.0 - eps)
    return np.log(u) - np.log1p(-u)

def log_sigmoid_prod(z):
    """
    Numerically stable ln(sigma(z) * (1 - sigma(z))) = -|z| - 2*ln(1 + exp(-|z|)).
    """
    abs_z = np.abs(np.asarray(z, dtype=float))
    return -abs_z - 2.0 * np.log1p(np.exp(-abs_z))


# =============================================================================
# Prior Range Accessor & Unconstrained Mappings
# =============================================================================

def get_prior_bounds(fittype):
    """
    Returns (N, 2) array of lower and upper prior bounds for a given model.
    """
    if fittype == 'twod_gaussring':
        return RING_PRIOR_RANGES
    elif fittype == 'twod_gauss1blob':
        return GAUSS1BLOB_PRIOR_RANGES
    else:
        raise ValueError(f"Unsupported fittype '{fittype}'. Must be 'twod_gaussring' or 'twod_gauss1blob'.")


def theta_to_zeta(theta, fittype):
    r"""
    Maps constrained physical parameters $\theta$ to unconstrained coordinates $\zeta \in \mathbb{R}^D$.
    """
    bounds = get_prior_bounds(fittype)
    theta = np.asarray(theta, dtype=float)
    low = bounds[:, 0]
    high = bounds[:, 1]
    ndim = len(low)

    zeta = np.empty(ndim, dtype=float)

    if fittype == 'twod_gaussring':
        u = (theta - low) / (high - low)
        zeta = logit(u)
    elif fittype == 'twod_gauss1blob':
        # Linear parameters: 0-8 and 10
        idx_lin = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
        u_lin = (theta[idx_lin] - low[idx_lin]) / (high[idx_lin] - low[idx_lin])
        zeta[idx_lin] = logit(u_lin)

        # Uniform area radial distance: parameter 9
        # r = sqrt(r_min^2 + u * (r_max^2 - r_min^2)) => u = (r^2 - r_min^2) / (r_max^2 - r_min^2)
        r = theta[9]
        u_r = (r**2 - low[9]**2) / (high[9]**2 - low[9]**2)
        zeta[9] = logit(u_r)

    return zeta


def zeta_to_theta(zeta, fittype):
    r"""
    Maps unconstrained coordinates $\zeta \in \mathbb{R}^D$ to constrained physical parameters $\theta$.
    """
    bounds = get_prior_bounds(fittype)
    zeta = np.asarray(zeta, dtype=float)
    low = bounds[:, 0]
    high = bounds[:, 1]
    ndim = len(low)

    u = sigmoid(zeta)

    if fittype == 'twod_gaussring':
        return low + u * (high - low)
    elif fittype == 'twod_gauss1blob':
        theta = np.empty(ndim, dtype=float)
        # Linear parameters: 0-8 and 10
        idx_lin = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
        theta[idx_lin] = low[idx_lin] + u[idx_lin] * (high[idx_lin] - low[idx_lin])

        # Radial distance with uniform area: parameter 9
        theta[9] = np.sqrt(low[9]**2 + u[9] * (high[9]**2 - low[9]**2))
        return theta


def grad_zeta_to_theta(zeta, fittype):
    r"""
    Computes partial derivatives $d\theta_i / d\zeta_i$ for each parameter.
    """
    bounds = get_prior_bounds(fittype)
    zeta = np.asarray(zeta, dtype=float)
    low = bounds[:, 0]
    high = bounds[:, 1]
    ndim = len(low)

    u = sigmoid(zeta)
    du_dzeta = u * (1.0 - u)

    if fittype == 'twod_gaussring':
        return (high - low) * du_dzeta
    elif fittype == 'twod_gauss1blob':
        dtheta_dzeta = np.empty(ndim, dtype=float)
        idx_lin = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
        dtheta_dzeta[idx_lin] = (high[idx_lin] - low[idx_lin]) * du_dzeta[idx_lin]

        # Parameter 9: theta[9] = sqrt(low^2 + u * (high^2 - low^2))
        # dtheta / du = (high^2 - low^2) / (2 * theta[9])
        r = np.sqrt(low[9]**2 + u[9] * (high[9]**2 - low[9]**2))
        dtheta_dzeta[9] = ((high[9]**2 - low[9]**2) / (2.0 * max(r, 1e-12))) * du_dzeta[9]
        return dtheta_dzeta


def log_prior_and_jacobian(zeta, fittype):
    r"""
    Evaluates the joint log-prior and transformation log-Jacobian determinant in unconstrained space:
        $\ln p(\zeta) = \ln p(\theta(\zeta)) + \sum_{i=1}^D \ln | \frac{\partial \theta_i}{\partial \zeta_i} | = \sum_{i=1}^D [\ln \sigma(\zeta_i) + \ln(1 - \sigma(\zeta_i))]$
    """
    return float(np.sum(log_sigmoid_prod(zeta)))


def grad_log_prior_and_jacobian(zeta, fittype):
    r"""
    Computes the exact analytic gradient of the joint prior and log-Jacobian with respect to $\zeta$:
        $\nabla_\zeta [\ln p(\theta(\zeta)) + \ln |J(\zeta)|] = 1 - 2\sigma(\zeta)$
    """
    return 1.0 - 2.0 * sigmoid(zeta)


def sample_prior(fittype, size=1):
    r"""
    Draws independent samples directly from the physical prior distribution.
    """
    bounds = get_prior_bounds(fittype)
    low = bounds[:, 0]
    high = bounds[:, 1]
    ndim = len(low)

    u = np.random.uniform(0.0, 1.0, size=(size, ndim))
    samples = np.empty((size, ndim), dtype=float)

    if fittype == 'twod_gaussring':
        samples = low + u * (high - low)
    elif fittype == 'twod_gauss1blob':
        idx_lin = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
        samples[:, idx_lin] = low[idx_lin] + u[:, idx_lin] * (high[idx_lin] - low[idx_lin])
        samples[:, 9] = np.sqrt(low[9]**2 + u[:, 9] * (high[9]**2 - low[9]**2))

    if size == 1:
        return samples[0]
    return samples
