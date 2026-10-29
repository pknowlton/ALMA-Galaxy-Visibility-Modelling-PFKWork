r"""
2D Galaxy Brightness Profile Models and Galario Visibility Interface
=====================================================================
This module defines parametric 2D intensity distributions (e.g. tilted Gaussian
rings with optional embedded clusters/blobs) representing the millimeter dust continuum
emission in galaxy nuclear rings (specifically applied to NGC 3351 / M95).

It interfaces with **Galario** (Gpu Accelerated Library for Analyzing Radio
Interferometer Observations), which computes synthetic Fourier visibilities
$V_{\text{mod}}(u, v)$ from simulated sky images and evaluates the goodness-of-fit
$\chi^2$ directly in the visibility domain:
    $\chi^2 = \sum_k w_k \left| V_{\text{obs}}(u_k, v_k) - V_{\text{mod}}(u_k, v_k) \right|^2$

Coordinate System Conventions (Galario `origin='lower'`):
---------------------------------------------------------
- The pixel array row index $iy \in [0, N_{xy}-1]$ corresponds to Declination offset $\Delta\delta$
  (increasing with row index, North is upward, $y > 0$).
- The pixel array column index $ix \in [0, N_{xy}-1]$ corresponds to Right Ascension offset $\Delta\alpha\cos\delta$
  (descending with column index, East is leftward, $x > 0$ at $ix = 0$, $x < 0$ at $ix = N_{xy}-1$).
- Physical coordinate origin is at pixel $[N_{xy}/2, N_{xy}/2]$.
- Disk Position Angle (PA) is defined East of North.
- All models use a consistent physical coordinate definition without hidden arbitrary rotation offsets.

Execution Modes & Angular Unit Handling (`version` argument):
-------------------------------------------------------------
All spatial parameters—including ring Gaussian sigma ($\sigma_{\text{ring}}$), blob Gaussian sigmas
($\sigma_{\text{blob}}$), ring radius ($R_{\text{ring}}$), centroid offsets
($\Delta\alpha, \Delta\delta$), and blob radial distances ($d_{\text{blob}}$)—can be
expressed in **radians** or **arcseconds**, depending on the execution mode:

1. `'chi2'` / `'vis'` (Fourier / Visibility domain):
   - Galario operates internally in **radians**.
   - Input parameters in `pars` are supplied in **arcseconds** by the sampler.
   - `evaluate_model` automatically converts all spatial dimensions from arcseconds to **radians**
     (via multiplication by `arcsec = np.pi / (180 * 3600)`).
   - Pixel scale `dxy` is provided in **radians** (from `get_image_size`).
   - The image builder and profile functions receive all spatial quantities
     consistently in **radians**.

2. `'plot'` (Sky Image / Visualization domain):
   - Pixel scale `dxy` is passed directly in **arcseconds** (from FITS header `CDELT2 * 3600`).
   - `evaluate_model` skips radian conversion, keeping all spatial parameters in **arcseconds**.
   - The image builder constructs coordinate grids $(xx, yy)$ in **arcseconds** to directly
     produce physical surface brightness maps in $\text{Jy}/\text{sr}$.

Model Structure:
----------------
What each `fittype` contains (ring or no ring, number of blobs, where the disk
geometry comes from, parameter labels) is defined once in `model_defs.py`.
All fittypes are evaluated by the same generic code (`evaluate_model`), so there is
no per-model wrapper function to keep in sync.
"""

import numpy as np
import logging
from galario.double import deg, arcsec, chi2Image, sampleImage
from model_defs import get_spec

# Module-level coordinate meshgrid cache to avoid heavy allocations across workers
_GRID_CACHE = {}

def get_grid(nxy, dxy):
    """
    Returns coordinate meshgrid (xx, yy) matching Galario's origin='lower' convention.

    xx : 2D array of Right Ascension offsets (East positive, descending with column index; East is left).
    yy : 2D array of Declination offsets (North positive, ascending with row index; North is up).

    Caches precomputed grids for each (nxy, dxy) pair to eliminate redundant memory allocations
    and significantly reduce per-likelihood evaluation overhead.
    """
    key = (int(nxy), float(dxy))
    if key not in _GRID_CACHE:
        # Galario convention: pixel [nxy/2, nxy/2] is (0, 0).
        # RA decreases with column index ix (East is left, ix=0 has x > 0).
        # Dec increases with row index iy (North is up, iy=0 has y < 0).
        x = -(np.arange(nxy) - nxy / 2.0) * dxy
        y = (np.arange(nxy) - nxy / 2.0) * dxy
        xx, yy = np.meshgrid(x, y)
        _GRID_CACHE[key] = (xx, yy)
    return _GRID_CACHE[key]

#########################
### Base profile functions
#########################

def gaussring_prof(peak_ring, sigma_ring, rad_ring, radius, dxy):
    r"""
    Computes the radial surface brightness of a Gaussian ring.

    The radial profile is defined as a 1D Gaussian centered at a specific ring radius:
        $I(r) = 10^{\text{peak\_ring}} \cdot \exp\left( -\frac{1}{2} \left(\frac{r - r_{\text{ring}}}{\sigma_{\text{ring}}}\right)^2 \right) \cdot \text{dxy}^2$

    Parameters
    ----------
    peak_ring : float
        Base-10 logarithm of the peak surface brightness in Jy/sr (i.e. $\log_{10}(I_0)$).
    sigma_ring : float
        Radial standard deviation (Gaussian sigma) of the ring. Note: FWHM = 2.35482 * sigma.
        Expressed in **radians** (for `'chi2'` / `'vis'`) or in **arcseconds** (for `'plot'`).
    rad_ring : float
        Central radial distance of the ring ridge from galaxy center.
        Expressed in **radians** (for `'chi2'` / `'vis'`) or in **arcseconds** (for `'plot'`).
    radius : numpy.ndarray
        2D grid of radial distances from the ring center, in matching units (**radians** or **arcseconds**).
    dxy : float
        Pixel scale (angular size per pixel in radians for `'chi2'`/`'vis'`, or 1 for direct surface brightness in `'plot'`).
        Multiplying by $\text{dxy}^2$ converts surface brightness into flux per pixel.

    Returns
    -------
    numpy.ndarray
        2D array of flux or intensity values for the ring component.
    """
    return 10**peak_ring * np.exp((-1/2)*((radius-rad_ring)/sigma_ring)**2) * (dxy**2)

def gaussring_nonuni_prof(peak_ring, theta, off_ring, sigma_ring, rad_ring, radius, dxy):
    r"""
    Computes the 2D surface brightness of a non-uniform Gaussian ring with m=2 azimuthal modulation.

    The azimuthal surface brightness is modeled as:
        $I(r, \theta) = 10^{\text{peak\_ring}} \cdot \left[1 + \cos\left(2 (\theta - \phi_{\text{off}})\right)\right] \cdot \exp\left( -\frac{1}{2} \left(\frac{r - r_{\text{ring}}}{\sigma_{\text{ring}}}\right)^2 \right) \cdot \text{dxy}^2$

    This formulation keeps the surface brightness non-negative everywhere ($I \ge 0$), creates two
    symmetric azimuthal peaks separated by 180 degrees at $\theta = \phi_{\text{off}}$ and $\theta = \phi_{\text{off}} + \pi$,
    and exactly preserves the integrated ring flux:
        $\int_0^{2\pi} [1 + \cos(2(\theta - \phi_{\text{off}}))] \, d\theta = 2\pi$
    Consequently, `ring_flux_to_peak(log_flux, sigma_rad, rad_rad, inc_rad)` remains mathematically
    exact for converting total integrated flux (in Jy) to `peak_ring` (log10 Jy/sr).

    Parameters
    ----------
    peak_ring : float
        Base-10 logarithm of the mean azimuthal peak surface brightness in Jy/sr (i.e. $\log_{10}(I_0)$).
        Peak surface brightness along the ring reaches $2 \cdot 10^{\text{peak\_ring}}$.
    theta : numpy.ndarray
        2D grid of azimuthal angles on the deprojected disk plane in **radians**, measured East of North.
    off_ring : float
        Azimuthal position angle of the emission peak on the disk in **radians**, measured East of North.
    sigma_ring : float
        Radial standard deviation (Gaussian sigma) of the ring. Note: FWHM = 2.35482 * sigma.
        Expressed in **radians** (for `'chi2'` / `'vis'`) or in **arcseconds** (for `'plot'`).
    rad_ring : float
        Central radial distance of the ring ridge from galaxy center.
        Expressed in **radians** (for `'chi2'` / `'vis'`) or in **arcseconds** (for `'plot'`).
    radius : numpy.ndarray
        2D grid of radial distances from the ring center, in matching units (**radians** or **arcseconds**).
    dxy : float
        Pixel scale (angular size per pixel in radians for `'chi2'`/`'vis'`, or 1 for direct surface brightness in `'plot'`).
        Multiplying by $\text{dxy}^2$ converts surface brightness into flux per pixel.

    Returns
    -------
    numpy.ndarray
        2D array of flux or intensity values for the non-uniform ring component.
    """
    azimuthal_factor = 1.0 + np.cos(2.0 * (theta - off_ring))
    radial_profile = np.exp((-0.5) * ((radius - rad_ring) / sigma_ring)**2)
    return (10.0**peak_ring) * azimuthal_factor * radial_profile * (dxy**2)

def gaussblob_prof(peak, sigma, xx, yy, xoff, yoff, dxy):
    r"""
    Computes the 2D surface brightness of a circular Gaussian cluster/blob.

    Represents localized emission peaks (e.g. starburst clusters / giant molecular clouds):
        $I(x, y) = 10^{\text{peak}} \cdot \exp\left( -\frac{1}{2} \left[ \left(\frac{x - x_{\text{off}}}{\sigma}\right)^2 + \left(\frac{y - y_{\text{off}}}{\sigma}\right)^2 \right] \right) \cdot \text{dxy}^2$

    Parameters
    ----------
    peak : float
        Base-10 logarithm of the cluster peak surface brightness in Jy/sr.
    sigma : float
        Gaussian standard deviation (cluster sigma). Note: FWHM = 2.35482 * sigma.
        Expressed in **radians** (for `'chi2'` / `'vis'`) or in **arcseconds** (for `'plot'`).
    xx : numpy.ndarray
        2D meshgrid array of X (RA) coordinates, in **radians** or **arcseconds** (East > 0).
    yy : numpy.ndarray
        2D meshgrid array of Y (Dec) coordinates, in **radians** or **arcseconds** (North > 0).
    xoff : float
        X-axis offset (RA, East > 0) of the cluster center, in matching units (**radians** or **arcseconds**).
    yoff : float
        Y-axis offset (Dec, North > 0) of the cluster center, in matching units (**radians** or **arcseconds**).
    dxy : float
        Pixel scale (angular size per pixel in radians for `'chi2'`/`'vis'`, or 1 for direct surface brightness in `'plot'`).

    Returns
    -------
    numpy.ndarray
        2D array of flux or intensity values for the cluster component.
    """
    return 10**peak * np.exp((-1/2) * (((xx-xoff)/sigma)**2 + ((yy-yoff)/sigma)**2)) * (dxy**2)

#########################
### Conversion functions
#########################

def ring_flux_to_peak(log_flux_jy, sigma_rad, rad_rad, inc_rad):
    """
    Converts log10 integrated flux of an inclined Gaussian ring to peak surface brightness log10(Jy/sr).

    Audit P1 Item 1:
        F_ring = (2*pi)**(3/2) * I_0 * R * sigma * cos(inc)
        where R and sigma are in radians, inc is in radians, F is in Jy, and I_0 is in Jy/sr.
    """
    flux_jy = 10.0**log_flux_jy
    cos_inc = max(float(np.cos(inc_rad)), 0.05)
    area_sr = ((2.0 * np.pi)**1.5) * float(rad_rad) * float(sigma_rad) * cos_inc
    i0_jysr = flux_jy / max(area_sr, 1e-30)
    return float(np.log10(max(i0_jysr, 1e-30)))

def blob_flux_to_peak(log_flux_jy, sigma_rad):
    """
    Converts log10 integrated flux of a circular 2D Gaussian blob to peak surface brightness log10(Jy/sr).

    Audit P1 Item 1:
        F_blob = 2 * pi * I_0 * sigma**2
        where sigma is in radians, F is in Jy, and I_0 is in Jy/sr.
    """
    flux_jy = 10.0**log_flux_jy
    area_sr = 2.0 * np.pi * (float(sigma_rad)**2)
    i0_jysr = flux_jy / max(area_sr, 1e-30)
    return float(np.log10(max(i0_jysr, 1e-30)))

#########################
### Generic model builder (ring + N blobs)
#########################
#
# Every fittype is "optional ring + N Gaussian blobs", as described in
# model_defs.py. The three functions below build any of them.
#
#   _split_pars        : pull ring / blob / geometry values out of the sampler's `pars`
#   _build_sky_image   : make the 2D image for 'chi2'/'vis' (radians) or 'plot' (arcsec)
#   evaluate_model     : image -> chi2, model visibilities, or plot image

def _split_pars(spec, pars):
    """
    Splits the flat parameter vector into ring, blob and geometry pieces,
    in the units the sampler uses (log10 Jy, log10 arcsec, arcsec, degrees).

    Returns
    -------
    ring : dict or None
        {'log_flux', 'log_sigma', 'rad', 'inc', 'offset'} ('offset' is None for a uniform ring)
    blobs : list of dict
        One {'log_flux', 'log_sigma', 'dist', 'ang'} per blob.
    geometry : tuple
        (posangle_deg, dRA_arcsec, dDec_arcsec)
    """
    if len(pars) != spec.ndim:
        msg = f"Model '{spec.name}' expects {spec.ndim} parameters, got {len(pars)}."
        logging.warning(msg)
        raise ValueError(msg)

    ring = None
    if spec.has_ring:
        ring = {
            'log_flux': pars[0], 'log_sigma': pars[1], 'rad': pars[2], 'inc': pars[3],
            'offset': pars[7] if spec.is_nonuni else None,
        }

    blobs = []
    for k in range(spec.n_blobs):
        i = spec.blob_index(k)
        blobs.append({'log_flux': pars[i], 'log_sigma': pars[i + 1], 'dist': pars[i + 2], 'ang': pars[i + 3]})

    return ring, blobs, spec.geometry_params(pars)


def _build_sky_image(ring, blobs, pa, dra, ddec, nxy, dxy, version):
    """
    Builds the 2D model image: ring (if any) + blobs, added in that order.

    `ring` and `blobs` hold the values listed below, already converted:
        ring  : peak, offset (rad or None), sigma, rad, inc (rad)
        blobs : peak, sigma, dist, ang (rad)

    Lengths (sigma, rad, dist, dra, ddec, dxy) are in **radians** for version in
    ('chi2', 'vis') and in **arcseconds** for 'plot'. Angles (inc, pa, ang, offset)
    are always radians.

    'chi2'/'vis': the image is built un-rotated and un-shifted; Galario applies
                  PA, dRA and dDec during the FFT.
    'plot'      : shift and rotation are applied here on the pixel grid, and the
                  result is a surface brightness in Jy/sr (dxy = 1).
    """
    xx, yy = get_grid(nxy, dxy)

    if version in ("chi2", "vis"):
        # Blob centres in the un-rotated frame (East: x > 0, North: y > 0)
        x, y = xx, yy
        pix = dxy
    elif version == 'plot':
        xx_shifted = xx - dra
        yy_shifted = yy - ddec
        x = xx_shifted * np.cos(pa) - yy_shifted * np.sin(pa)
        y = xx_shifted * np.sin(pa) + yy_shifted * np.cos(pa)
        pix = 1
    else:
        msg = f"Invalid version '{version}', must be 'chi2', 'vis', or 'plot'"
        logging.warning(msg)
        raise ValueError(msg)

    image = None

    if ring is not None:
        # Deproject x-axis by cos(inclination) to account for disk tilt
        xinc = x / np.cos(ring['inc'])
        radius_vec = np.hypot(xinc, y)
        if ring['offset'] is None:
            image = gaussring_prof(ring['peak'], ring['sigma'], ring['rad'], radius_vec, pix)
        else:
            # Azimuthal angle in the disk plane (East of North)
            theta_vec = np.arctan2(xinc, y)
            image = gaussring_nonuni_prof(ring['peak'], theta_vec, ring['offset'], ring['sigma'], ring['rad'], radius_vec, pix)

    for b in blobs:
        # Calculate Cartesian offsets (East: xdot > 0, North: ydot > 0)
        xdot = b['dist'] * np.sin(b['ang'])
        ydot = b['dist'] * np.cos(b['ang'])
        blob_img = gaussblob_prof(b['peak'], b['sigma'], x, y, xdot, ydot, pix)
        image = blob_img if image is None else image + blob_img

    return image


def evaluate_model(spec, pars, args, vis_data, version):
    """
    Evaluates any model described by a ModelSpec.

    Converts parameter units (degrees to radians; arcseconds to radians when fitting),
    reparameterizes log flux and log sigma to physical peak surface brightness,
    builds the 2D image, and returns chi^2, model visibilities, or the image.

    Parameters
    ----------
    spec : model_defs.ModelSpec
    pars : array_like
        Parameter vector in the order given by `spec.labels_units()`.
    args : tuple
        `(nxy, dxy)`. `dxy` is in **radians** for 'chi2'/'vis', **arcseconds** for 'plot'.
    vis_data : tuple
        (u, v, Re, Im, weights).
    version : str
        `'chi2'`, `'vis'` or `'plot'`.

    Returns
    -------
    float or numpy.ndarray
        - 'chi2': scalar chi^2.
        - 'vis' : 1D complex array of model visibilities.
        - 'plot': 2D array of model surface brightness (Jy/sr).
    """
    ring_p, blob_p, (posangle, dRA, dDec) = _split_pars(spec, pars)
    nxy, dxy = args
    u, v, re, im, w = vis_data

    # Lengths are converted to radians when fitting and left in arcsec for plots
    to_len = (lambda x: x * arcsec) if version in ("chi2", "vis") else (lambda x: x)

    posangle_rad = posangle * deg

    ring = None
    if ring_p is not None:
        inc_rad = ring_p['inc'] * deg
        sigma_arcsec = 10.0**ring_p['log_sigma']
        sigma_rad = sigma_arcsec * arcsec
        rad_rad = ring_p['rad'] * arcsec
        ring = {
            'peak': ring_flux_to_peak(ring_p['log_flux'], sigma_rad, rad_rad, inc_rad),
            'offset': None if ring_p['offset'] is None else ring_p['offset'] * deg,
            'sigma': to_len(sigma_arcsec),
            'rad': to_len(ring_p['rad']),
            'inc': inc_rad,
        }

    blobs = []
    for b in blob_p:
        sigma_arcsec = 10.0**b['log_sigma']
        sigma_rad = sigma_arcsec * arcsec
        blobs.append({
            'peak': blob_flux_to_peak(b['log_flux'], sigma_rad),
            'sigma': to_len(sigma_arcsec),
            'dist': to_len(b['dist']),
            'ang': b['ang'] * deg,
        })

    model_img = _build_sky_image(ring, blobs, posangle_rad, to_len(dRA), to_len(dDec), nxy, dxy, version)

    if version == 'plot':
        return model_img

    dRA_rad = dRA * arcsec
    dDec_rad = dDec * arcsec
    if version == 'chi2':
        return chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA_rad, dDec=dDec_rad, PA=posangle_rad, origin='lower')
    return np.array(sampleImage(model_img, dxy, u, v, dRA=dRA_rad, dDec=dDec_rad, PA=posangle_rad, origin='lower'), dtype=np.complex256)


##################################################################################################################################################

def model_prof(pars, args, vis_data, version, fittype):
    """
    Central dispatcher function for model profile evaluation.

    The structure of each `fittype` (ring, number of blobs, geometry) is defined
    in `model_defs.py`.
    """
    if version not in ("chi2", "vis", "plot"):
        msg = f"Invalid version '{version}', must be 'chi2', 'vis', or 'plot'"
        logging.warning(msg)
        raise ValueError(msg)

    return evaluate_model(get_spec(fittype), pars, args, vis_data, version)

##################################################################################################################################################

def model_addon(fittype):
    """
    Returns parameter metadata (labels, physical units, dimensionality) for a given model.
    Reparameterized according to Audit P1 Item 1: log integrated flux + log size.
    """
    spec = get_spec(fittype)
    label, unit = spec.labels_units()
    return label, unit, len(label)
