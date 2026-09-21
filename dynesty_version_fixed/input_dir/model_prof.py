"""
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
   - The top-level wrapper functions (`twod_gaussring`, `twod_gauss1blob`, `twod_gauss1blob_2peak`, `twod_gauss1blob_2peak_dp`, `twod_gauss3blob`)
     automatically convert all spatial dimensions from arcseconds to **radians**
     (via multiplication by `arcsec = np.pi / (180 * 3600)`).
   - Pixel scale `dxy` is provided in **radians** (from `get_image_size`).
   - The low-level model generators and profile functions receive all spatial quantities
     consistently in **radians**.

2. `'plot'` (Sky Image / Visualization domain):
   - Pixel scale `dxy` is passed directly in **arcseconds** (from FITS header `CDELT2 * 3600`).
   - The top-level wrappers skip radian conversion, keeping all spatial parameters in **arcseconds**.
   - The model generators construct coordinate grids $(xx, yy)$ in **arcseconds** to directly
     produce physical surface brightness maps in $\text{Jy}/\text{sr}$.
"""

import numpy as np
import logging
from galario.double import get_image_size, deg, arcsec, chi2Profile, chi2Image, sampleProfile, sampleImage
import pandas as pd

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
    """
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

def gaussblob_prof(peak, sigma, xx, yy, xoff, yoff, dxy):
    """
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

def gaussblob_2peak_prof(peak1, sigma1, peak2, sigma2, xx, yy, xoff, yoff, dxy):
    """
    Computes the 2D surface brightness of a two-component (composite) Gaussian cluster/blob.

    Represents a localized emission knot modeled as the sum of two concentric Gaussian
    components with distinct peak intensities and standard deviations (sigmas):
        $I(x, y) = \left( 10^{\text{peak}_1} \cdot \exp\left( -\frac{1}{2} \left[ \left(\frac{x - x_{\text{off}}}{\sigma_1}\right)^2 + \left(\frac{y - y_{\text{off}}}{\sigma_1}\right)^2 \right] \right) + 10^{\text{peak}_2} \cdot \exp\left( -\frac{1}{2} \left[ \left(\frac{x - x_{\text{off}}}{\sigma_2}\right)^2 + \left(\frac{y - y_{\text{off}}}{\sigma_2}\right)^2 \right] \right) \right) \cdot \text{dxy}^2$

    Parameters
    ----------
    peak1 : float
        Base-10 logarithm of the first component peak surface brightness in Jy/sr.
    sigma1 : float
        Gaussian standard deviation (sigma) of the first component.
    peak2 : float
        Base-10 logarithm of the second component peak surface brightness in Jy/sr.
    sigma2 : float
        Gaussian standard deviation (sigma) of the second component.
    xx : numpy.ndarray
        2D meshgrid array of X coordinates, in **radians** or **arcseconds**.
    yy : numpy.ndarray
        2D meshgrid array of Y coordinates, in **radians** or **arcseconds**.
    xoff : float
        X-axis offset of the composite cluster center from origin, in matching units (**radians** or **arcseconds**).
    yoff : float
        Y-axis offset of the composite cluster center from origin, in matching units (**radians** or **arcseconds**).
    dxy : float
        Pixel scale (angular size per pixel in radians for `'chi2'`/`'vis'`, or 1 for direct surface brightness in `'plot'`).

    Returns
    -------
    numpy.ndarray
        2D array of flux or intensity values for the two-component cluster.
    """
    g1 = 10**peak1 * np.exp((-1/2) * (((xx-xoff)/sigma1)**2 + ((yy-yoff)/sigma1)**2)) * (dxy**2)
    g2 = 10**peak2 * np.exp((-1/2) * (((xx-xoff)/sigma2)**2 + ((yy-yoff)/sigma2)**2)) * (dxy**2)
    
    return g1 + g2

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
### 2D Gaussian Ring
#########################

def twod_gaussring_model(peak, sigma, rad, inc, pa, dra, ddec, nxy, dxy, version):
    """
    Generates a 2D image matrix representing an inclined, rotated Gaussian ring.

    Parameters
    ----------
    peak : float
        Log10 of peak surface brightness [log(Jy/sr)].
    sigma : float
        Gaussian radial sigma. In **radians** for `version in ('chi2', 'vis')`; in **arcseconds** for `version == 'plot'`.
    rad : float
        Central ring radius. In **radians** for `version in ('chi2', 'vis')`; in **arcseconds** for `version == 'plot'`.
    inc : float
        Disk inclination angle in radians (0 = face-on, $\pi/2$ = edge-on).
    pa : float
        Disk position angle in radians (East of North).
    dra : float
        Right Ascension center offset. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    ddec : float
        Declination center offset. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    nxy : int
        Number of pixels along each axis of the square image grid.
    dxy : float
        Angular pixel scale. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    version : str
        Operating mode:
            - `'chi2'` / `'vis'`: Computes deprojected image in intrinsic coordinates (radians);
                                  spatial shifts ($\Delta\alpha, \Delta\delta$) and rotation
                                  are handled analytically by Galario during FFT.
            - `'plot'`: Applies spatial shifts and rotation directly on the pixel grid (arcseconds)
                        for displaying in sky coordinates.

    Returns
    -------
    numpy.ndarray
        2D numpy array of shape (nxy, nxy) containing model sky brightness.
    """
    xx, yy = get_grid(nxy, dxy)

    if version in ("chi2", "vis"):
        # Deproject x-axis by cos(inclination) to account for disk tilt (all units in radians)
        xinc = xx / np.cos(inc)
        radius_vec = np.hypot(xinc, yy)

        ring_model_jypix = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
        return ring_model_jypix

    elif version == 'plot':
        # Shift coordinates to account for centroid offsets (all units in arcseconds)
        xx_shifted = xx - dra
        yy_shifted = yy - ddec

        # Rotate coordinates by position angle (PA, East of North)
        xpa =  xx_shifted * np.cos(pa) - yy_shifted * np.sin(pa)
        ypa =  xx_shifted * np.sin(pa) + yy_shifted * np.cos(pa)

        # Deproject inclined circular ring into an ellipse
        xinc = xpa / np.cos(inc)
        radius_vec = np.hypot(xinc, ypa)

        # Return surface brightness in Jy/sr (dxy = 1)
        ring_model_jysr = gaussring_prof(peak, sigma, rad, radius_vec, 1)
        return ring_model_jysr

    else:
        msg = f"Invalid version '{version}', must be 'chi2', 'vis', or 'plot'"
        logging.warning(msg)
        raise ValueError(msg)

def twod_gaussring(pars, args, vis_data, version):
    """
    Top-level model handler for the 2D Gaussian Ring.

    Converts parameter units (degrees to radians; arcseconds to radians when fitting),
    reparameterizes log flux and log sigma to physical peak surface brightness,
    constructs the 2D image model, and evaluates chi^2, model visibilities, or image maps.

    Parameters
    ----------
    pars : array_like
        Model parameter vector (Audit P1 Item 1: log integrated flux + log size):
        [log_flux (log10 Jy), log_sigma (log10 arcsec), ring_rad (arcsec), inclination (deg),
         posangle (deg), dRA (arcsec), dDec (arcsec)]
    args : tuple
        Image dimension parameters: `(nxy, dxy)`:
        - `dxy` is in **radians** for `version in ('chi2', 'vis')`
        - `dxy` is in **arcseconds** for `version == 'plot'`
    vis_data : tuple
        Interferometric visibility data: (u, v, Re, Im, weights).
    version : str
        Evaluation mode: `'chi2'`, `'vis'`, or `'plot'`.

    Returns
    -------
    float or numpy.ndarray
        - If `version == 'chi2'`: Scalar goodness-of-fit chi^2 value.
        - If `version == 'vis'`: 1D complex numpy array of sampled model visibilities.
        - If `version == 'plot'`: 2D real numpy array of model image brightness.
    """
    log_flux, log_sigma, ring_rad, inclination, posangle, dRA, dDec = pars
    nxy, dxy = args
    u, v, re, im, w = vis_data

    inclination *= deg
    posangle *= deg

    sigma_arcsec = 10.0**log_sigma
    sigma_rad = sigma_arcsec * arcsec
    ring_rad_rad = ring_rad * arcsec
    peak_ring = ring_flux_to_peak(log_flux, sigma_rad, ring_rad_rad, inclination)

    if version in ("chi2", "vis"):
        dRA_rad = dRA * arcsec
        dDec_rad = dDec * arcsec
        model_img = twod_gaussring_model(peak_ring, sigma_rad, ring_rad_rad, inclination, posangle, dRA_rad, dDec_rad, nxy, dxy, version) 

        if version == 'chi2':
            chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower')
            return chi2
        else:
            model_vis = np.array(sampleImage(model_img, dxy, u, v, dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower'), dtype=np.complex256)
            return model_vis
    else:
        model_img = twod_gaussring_model(peak_ring, sigma_arcsec, ring_rad, inclination, posangle, dRA, dDec, nxy, dxy, version)
        return model_img

#########################
### 2D Gaussian Ring + 1 Gaussian Blob
#########################

def twod_gauss1blob_model(peak, sigma, rad, inc, pa, dra, ddec, peak_b1, sigma_b1, dist_b1, ang_b1, nxy, dxy, version):
    """
    Generates a 2D image matrix of an inclined Gaussian ring plus 1 compact Gaussian cluster.

    Parameters
    ----------
    peak : float
        Ring peak brightness [log10(Jy/sr)].
    sigma, rad : float
        Ring radial sigma and central radius. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    inc, pa : float
        Disk inclination and position angle [rad].
    dra, ddec : float
        Centroid offsets in RA and Dec. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    peak_b1 : float
        Blob 1 peak brightness [log10(Jy/sr)].
    sigma_b1, dist_b1 : float
        Blob 1 Gaussian sigma and radial distance from ring center. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    ang_b1 : float
        Blob 1 azimuthal angle [rad] (East of North in disk frame).
    nxy : int
        Grid pixel count along each dimension.
    dxy : float
        Angular pixel scale. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    version : str
        Evaluation mode (`'chi2'`, `'vis'`, or `'plot'`).

    Returns
    -------
    numpy.ndarray
        2D array representing combined ring and blob intensity.
    """
    xx, yy = get_grid(nxy, dxy)

    # Calculate Cartesian offsets for Blob 1 (East: xdot > 0, North: ydot > 0)
    xdot_b1 = dist_b1 * np.sin(ang_b1)
    ydot_b1 = dist_b1 * np.cos(ang_b1)

    if version in ("chi2", "vis"):
        # Deproject x-axis by cos(inclination) (all spatial units in radians)
        xinc = xx / np.cos(inc)
        radius_vec = np.hypot(xinc, yy)

        ring_model_jypix = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
        blob1_model_jypix = gaussblob_prof(peak_b1, sigma_b1, xx, yy, xdot_b1, ydot_b1, dxy)

        return ring_model_jypix + blob1_model_jypix

    elif version == 'plot':
        # Shift and rotate coordinates (all spatial units in arcseconds)
        xx_shifted = xx - dra
        yy_shifted = yy - ddec

        xpa =  xx_shifted * np.cos(pa) - yy_shifted * np.sin(pa)
        ypa =  xx_shifted * np.sin(pa) + yy_shifted * np.cos(pa)

        xinc = xpa / np.cos(inc)
        radius_vec = np.hypot(xinc, ypa)

        ring_model_jysr = gaussring_prof(peak, sigma, rad, radius_vec, 1)
        blob_model_jysr = gaussblob_prof(peak_b1, sigma_b1, xpa, ypa, xdot_b1, ydot_b1, 1)

        return ring_model_jysr + blob_model_jysr

    else:
        msg = f"Invalid version '{version}', must be 'chi2', 'vis', or 'plot'"
        logging.warning(msg)
        raise ValueError(msg)

def twod_gauss1blob(pars, args, vis_data, version):
    """
    Top-level model handler for 2D Gaussian Ring + 1 Gaussian Blob (11 parameters).

    Parameters
    ----------
    pars : array_like (Audit P1 Item 1: log integrated flux + log size)
        [log_flux, log_sigma, ring_rad, inclination, posangle, dRA, dDec,
         log_flux_b1, log_sigma_b1, dist_b1, ang_b1]
    args : tuple
        `(nxy, dxy)`
    vis_data : tuple
        (u, v, Re, Im, weights)
    version : str
        `'chi2'`, `'vis'`, or `'plot'`
    """
    log_flux, log_sigma, ring_rad, inclination, posangle, dRA, dDec, log_flux_b1, log_sigma_b1, dist_b1, ang_b1 = pars
    nxy, dxy = args
    u, v, re, im, w = vis_data

    inclination *= deg
    posangle *= deg
    ang_b1 *= deg

    sigma_arcsec = 10.0**log_sigma
    sigma_b1_arcsec = 10.0**log_sigma_b1

    sigma_rad = sigma_arcsec * arcsec
    sigma_b1_rad = sigma_b1_arcsec * arcsec
    ring_rad_rad = ring_rad * arcsec

    peak_ring = ring_flux_to_peak(log_flux, sigma_rad, ring_rad_rad, inclination)
    peak_b1 = blob_flux_to_peak(log_flux_b1, sigma_b1_rad)

    if version in ("chi2", "vis"):
        dRA_rad = dRA * arcsec
        dDec_rad = dDec * arcsec
        dist_b1_rad = dist_b1 * arcsec
        model_img = twod_gauss1blob_model(peak_ring, sigma_rad, ring_rad_rad, inclination, posangle, dRA_rad, dDec_rad, peak_b1, sigma_b1_rad, dist_b1_rad, ang_b1, nxy, dxy, version) 

        if version == 'chi2':
            chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower')
            return chi2
        else:
            model_vis = np.array(sampleImage(model_img, dxy, u, v, dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower'), dtype=np.complex256)
            return model_vis
    else:
        model_img = twod_gauss1blob_model(peak_ring, sigma_arcsec, ring_rad, inclination, posangle, dRA, dDec, peak_b1, sigma_b1_arcsec, dist_b1, ang_b1, nxy, dxy, version)
        return model_img

#########################
### 2D Gaussian Ring + 1 Gaussian Blob (2 Concentric Peaks)
#########################

def twod_gauss1blob_2peak_model(peak, sigma, rad, inc, pa, dra, ddec, peak_b11, sigma_b11, peak_b12, sigma_b12, dist_b1, ang_b1, nxy, dxy, version):
    """
    Generates a 2D image matrix of an inclined Gaussian ring plus 1 composite 2-peak Gaussian cluster.
    """
    xx, yy = get_grid(nxy, dxy)

    # Calculate Cartesian offsets for Blob 1 (East: xdot > 0, North: ydot > 0)
    xdot_b1 = dist_b1 * np.sin(ang_b1)
    ydot_b1 = dist_b1 * np.cos(ang_b1)

    if version in ("chi2", "vis"):
        xinc = xx / np.cos(inc)
        radius_vec = np.hypot(xinc, yy)

        ring_model_jypix = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
        blob1_model_jypix = gaussblob_2peak_prof(peak_b11, sigma_b11, peak_b12, sigma_b12, xx, yy, xdot_b1, ydot_b1, dxy)

        return ring_model_jypix + blob1_model_jypix

    elif version == 'plot':
        xx_shifted = xx - dra
        yy_shifted = yy - ddec

        xpa =  xx_shifted * np.cos(pa) - yy_shifted * np.sin(pa)
        ypa =  xx_shifted * np.sin(pa) + yy_shifted * np.cos(pa)

        xinc = xpa / np.cos(inc)
        radius_vec = np.hypot(xinc, ypa)

        ring_model_jysr = gaussring_prof(peak, sigma, rad, radius_vec, 1)
        blob_model_jysr = gaussblob_2peak_prof(peak_b11, sigma_b11, peak_b12, sigma_b12, xpa, ypa, xdot_b1, ydot_b1, 1)

        return ring_model_jysr + blob_model_jysr

    else:
        msg = f"Invalid version '{version}', must be 'chi2', 'vis', or 'plot'"
        logging.warning(msg)
        raise ValueError(msg)

def twod_gauss1blob_2peak(pars, args, vis_data, version):
    """
    Top-level model handler for 2D Gaussian Ring + 1 Gaussian Blob with 2 Peaks (13 parameters).
    """
    log_flux, log_sigma, ring_rad, inclination, posangle, dRA, dDec, log_flux_b11, log_sigma_b11, log_flux_b12, log_sigma_b12, dist_b1, ang_b1 = pars
    nxy, dxy = args
    u, v, re, im, w = vis_data

    inclination *= deg
    posangle *= deg
    ang_b1 *= deg

    sigma_arcsec = 10.0**log_sigma
    sigma_b11_arcsec = 10.0**log_sigma_b11
    sigma_b12_arcsec = 10.0**log_sigma_b12

    sigma_rad = sigma_arcsec * arcsec
    sigma_b11_rad = sigma_b11_arcsec * arcsec
    sigma_b12_rad = sigma_b12_arcsec * arcsec
    ring_rad_rad = ring_rad * arcsec

    peak_ring = ring_flux_to_peak(log_flux, sigma_rad, ring_rad_rad, inclination)
    peak_b11 = blob_flux_to_peak(log_flux_b11, sigma_b11_rad)
    peak_b12 = blob_flux_to_peak(log_flux_b12, sigma_b12_rad)

    if version in ("chi2", "vis"):
        dRA_rad = dRA * arcsec
        dDec_rad = dDec * arcsec
        dist_b1_rad = dist_b1 * arcsec
        model_img = twod_gauss1blob_2peak_model(peak_ring, sigma_rad, ring_rad_rad, inclination, posangle, dRA_rad, dDec_rad, peak_b11, sigma_b11_rad, peak_b12, sigma_b12_rad, dist_b1_rad, ang_b1, nxy, dxy, version) 

        if version == 'chi2':
            chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower')
            return chi2
        else:
            model_vis = np.array(sampleImage(model_img, dxy, u, v, dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower'), dtype=np.complex256)
            return model_vis
    else:
        model_img = twod_gauss1blob_2peak_model(peak_ring, sigma_arcsec, ring_rad, inclination, posangle, dRA, dDec, peak_b11, sigma_b11_arcsec, peak_b12, sigma_b12_arcsec, dist_b1, ang_b1, nxy, dxy, version)
        return model_img

#########################
### 2D Gaussian Ring + 1 Gaussian Blob (2 Peaks, Double Pendulum)
#########################

def twod_gauss1blob_2peak_dp_model(peak, sigma, rad, inc, pa, dra, ddec, peak_b11, sigma_b11, dist_b11, ang_b11, peak_b12, sigma_b12, dist_b12, ang_b12, nxy, dxy, version):
    """
    Constructs a 2D intensity grid for a tilted Gaussian ring with a 2-peak "Double Pendulum" clump.

    Peak 1 is positioned at polar coordinates (dist_b11, ang_b11) relative to the galaxy center.
    Peak 2 is positioned at relative polar coordinates (dist_b12, ang_b12) with respect to Peak 1.
    """
    xx, yy = get_grid(nxy, dxy)

    # Calculate Cartesian offsets for Blob 1 Peak 1 (East: xdot > 0, North: ydot > 0)
    xdot_b11 = dist_b11 * np.sin(ang_b11)
    ydot_b11 = dist_b11 * np.cos(ang_b11)

    # Calculate extra Cartesian offsets for Blob 1 Peak 2 (relative to Peak 1)
    xdot_b12 = dist_b12 * np.sin(ang_b12)
    ydot_b12 = dist_b12 * np.cos(ang_b12)
    xdot_bdp = xdot_b11 + xdot_b12
    ydot_bdp = ydot_b11 + ydot_b12

    if version in ("chi2", "vis"):
        xinc = xx / np.cos(inc)
        radius_vec = np.hypot(xinc, yy)

        ring_model_jypix = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
        blob11_model_jypix = gaussblob_prof(peak_b11, sigma_b11, xx, yy, xdot_b11, ydot_b11, dxy)
        blob12_model_jypix = gaussblob_prof(peak_b12, sigma_b12, xx, yy, xdot_bdp, ydot_bdp, dxy)

        return ring_model_jypix + blob11_model_jypix + blob12_model_jypix

    elif version == 'plot':
        xx_shifted = xx - dra
        yy_shifted = yy - ddec

        xpa =  xx_shifted * np.cos(pa) - yy_shifted * np.sin(pa)
        ypa =  xx_shifted * np.sin(pa) + yy_shifted * np.cos(pa)

        xinc = xpa / np.cos(inc)
        radius_vec = np.hypot(xinc, ypa)

        ring_model_jysr = gaussring_prof(peak, sigma, rad, radius_vec, 1)
        blob11_model_jysr = gaussblob_prof(peak_b11, sigma_b11, xpa, ypa, xdot_b11, ydot_b11, 1)
        blob12_model_jysr = gaussblob_prof(peak_b12, sigma_b12, xpa, ypa, xdot_bdp, ydot_bdp, 1)

        return ring_model_jysr + blob11_model_jysr + blob12_model_jysr

    else:
        msg = f"Invalid version '{version}', must be 'chi2', 'vis', or 'plot'"
        logging.warning(msg)
        raise ValueError(msg)

def twod_gauss1blob_2peak_dp(pars, args, vis_data, version):
    """
    Top-level model handler for 2D Gaussian Ring + 1 Gaussian Blob with 2 Peaks (Double Pendulum, 15 parameters).
    """
    log_flux, log_sigma, ring_rad, inclination, posangle, dRA, dDec, log_flux_b11, log_sigma_b11, dist_b11, ang_b11, log_flux_b12, log_sigma_b12, dist_b12, ang_b12 = pars
    nxy, dxy = args
    u, v, re, im, w = vis_data

    inclination *= deg
    posangle *= deg
    ang_b11 *= deg
    ang_b12 *= deg

    sigma_arcsec = 10.0**log_sigma
    sigma_b11_arcsec = 10.0**log_sigma_b11
    sigma_b12_arcsec = 10.0**log_sigma_b12

    sigma_rad = sigma_arcsec * arcsec
    sigma_b11_rad = sigma_b11_arcsec * arcsec
    sigma_b12_rad = sigma_b12_arcsec * arcsec
    ring_rad_rad = ring_rad * arcsec

    peak_ring = ring_flux_to_peak(log_flux, sigma_rad, ring_rad_rad, inclination)
    peak_b11 = blob_flux_to_peak(log_flux_b11, sigma_b11_rad)
    peak_b12 = blob_flux_to_peak(log_flux_b12, sigma_b12_rad)

    if version in ("chi2", "vis"):
        dRA_rad = dRA * arcsec
        dDec_rad = dDec * arcsec
        dist_b11_rad = dist_b11 * arcsec
        dist_b12_rad = dist_b12 * arcsec
        model_img = twod_gauss1blob_2peak_dp_model(peak_ring, sigma_rad, ring_rad_rad, inclination, posangle, dRA_rad, dDec_rad, peak_b11, sigma_b11_rad, dist_b11_rad, ang_b11, peak_b12, sigma_b12_rad, dist_b12_rad, ang_b12, nxy, dxy, version) 

        if version == 'chi2':
            chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower')
            return chi2
        else:
            model_vis = np.array(sampleImage(model_img, dxy, u, v, dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower'), dtype=np.complex256)
            return model_vis
    else:
        model_img = twod_gauss1blob_2peak_dp_model(peak_ring, sigma_arcsec, ring_rad, inclination, posangle, dRA, dDec, peak_b11, sigma_b11_arcsec, dist_b11, ang_b11, peak_b12, sigma_b12_arcsec, dist_b12, ang_b12, nxy, dxy, version)
        return model_img

#########################
### 2D Gaussian Ring + 3 Gaussian Blobs
#########################

def twod_gauss3blob_model(peak, sigma, rad, inc, pa, dra, ddec, peak_b1, sigma_b1, dist_b1, ang_b1, peak_b2, sigma_b2, dist_b2, ang_b2, peak_b3, sigma_b3, dist_b3, ang_b3, nxy, dxy, version):
    """
    Generates a 2D image matrix of an inclined Gaussian ring plus 3 distinct Gaussian clusters.
    """
    xx, yy = get_grid(nxy, dxy)

    xdot_b1 = dist_b1 * np.sin(ang_b1)
    ydot_b1 = dist_b1 * np.cos(ang_b1)

    xdot_b2 = dist_b2 * np.sin(ang_b2)
    ydot_b2 = dist_b2 * np.cos(ang_b2)

    xdot_b3 = dist_b3 * np.sin(ang_b3)
    ydot_b3 = dist_b3 * np.cos(ang_b3)

    if version in ("chi2", "vis"):
        xinc = xx / np.cos(inc)
        radius_vec = np.hypot(xinc, yy)

        ring_model_jypix = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
        blob1_model_jypix = gaussblob_prof(peak_b1, sigma_b1, xx, yy, xdot_b1, ydot_b1, dxy)
        blob2_model_jypix = gaussblob_prof(peak_b2, sigma_b2, xx, yy, xdot_b2, ydot_b2, dxy)
        blob3_model_jypix = gaussblob_prof(peak_b3, sigma_b3, xx, yy, xdot_b3, ydot_b3, dxy)

        return ring_model_jypix + blob1_model_jypix + blob2_model_jypix + blob3_model_jypix

    elif version == 'plot':
        xx_shifted = xx - dra
        yy_shifted = yy - ddec

        xpa =  xx_shifted * np.cos(pa) - yy_shifted * np.sin(pa)
        ypa =  xx_shifted * np.sin(pa) + yy_shifted * np.cos(pa)

        xinc = xpa / np.cos(inc)
        radius_vec = np.hypot(xinc, ypa)

        ring_model_jysr = gaussring_prof(peak, sigma, rad, radius_vec, 1)
        blob1_model_jysr = gaussblob_prof(peak_b1, sigma_b1, xpa, ypa, xdot_b1, ydot_b1, 1)
        blob2_model_jysr = gaussblob_prof(peak_b2, sigma_b2, xpa, ypa, xdot_b2, ydot_b2, 1)
        blob3_model_jysr = gaussblob_prof(peak_b3, sigma_b3, xpa, ypa, xdot_b3, ydot_b3, 1)

        return ring_model_jysr + blob1_model_jysr + blob2_model_jysr + blob3_model_jysr

    else:
        msg = f"Invalid version '{version}', must be 'chi2', 'vis', or 'plot'"
        logging.warning(msg)
        raise ValueError(msg)

def twod_gauss3blob(pars, args, vis_data, version):
    """
    Top-level model handler for 2D Gaussian Ring + 3 Gaussian Blobs (19 parameters).
    """
    log_flux, log_sigma, ring_rad, inclination, posangle, dRA, dDec, log_flux_b1, log_sigma_b1, dist_b1, ang_b1, log_flux_b2, log_sigma_b2, dist_b2, ang_b2, log_flux_b3, log_sigma_b3, dist_b3, ang_b3 = pars
    nxy, dxy = args
    u, v, re, im, w = vis_data

    inclination *= deg
    posangle *= deg
    ang_b1 *= deg
    ang_b2 *= deg
    ang_b3 *= deg

    sigma_arcsec = 10.0**log_sigma
    sigma_b1_arcsec = 10.0**log_sigma_b1
    sigma_b2_arcsec = 10.0**log_sigma_b2
    sigma_b3_arcsec = 10.0**log_sigma_b3

    sigma_rad = sigma_arcsec * arcsec
    sigma_b1_rad = sigma_b1_arcsec * arcsec
    sigma_b2_rad = sigma_b2_arcsec * arcsec
    sigma_b3_rad = sigma_b3_arcsec * arcsec
    ring_rad_rad = ring_rad * arcsec

    peak_ring = ring_flux_to_peak(log_flux, sigma_rad, ring_rad_rad, inclination)
    peak_b1 = blob_flux_to_peak(log_flux_b1, sigma_b1_rad)
    peak_b2 = blob_flux_to_peak(log_flux_b2, sigma_b2_rad)
    peak_b3 = blob_flux_to_peak(log_flux_b3, sigma_b3_rad)

    if version in ("chi2", "vis"):
        dRA_rad = dRA * arcsec
        dDec_rad = dDec * arcsec
        dist_b1_rad = dist_b1 * arcsec
        dist_b2_rad = dist_b2 * arcsec
        dist_b3_rad = dist_b3 * arcsec
        model_img = twod_gauss3blob_model(peak_ring, sigma_rad, ring_rad_rad, inclination, posangle, dRA_rad, dDec_rad, peak_b1, sigma_b1_rad, dist_b1_rad, ang_b1, peak_b2, sigma_b2_rad, dist_b2_rad, ang_b2, peak_b3, sigma_b3_rad, dist_b3_rad, ang_b3, nxy, dxy, version) 

        if version == 'chi2':
            chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower')
            return chi2
        else:
            model_vis = np.array(sampleImage(model_img, dxy, u, v, dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower'), dtype=np.complex256)
            return model_vis
    else:
        model_img = twod_gauss3blob_model(peak_ring, sigma_arcsec, ring_rad, inclination, posangle, dRA, dDec, peak_b1, sigma_b1_arcsec, dist_b1, ang_b1, peak_b2, sigma_b2_arcsec, dist_b2, ang_b2, peak_b3, sigma_b3_arcsec, dist_b3, ang_b3, nxy, dxy, version)
        return model_img

#########################
### Simulated Single Gaussian Blob (7 parameters)
#########################

def twod_simgauss_model(peak, sigma, dra, ddec, pa, dist, ang, nxy, dxy, version):
    """
    Generates a 2D image matrix of a single compact Gaussian blob.

    Parameters
    ----------
    peak : float
        Blob peak brightness [log10(Jy/sr)].
    sigma : float
        Blob Gaussian sigma. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    dra, ddec : float
        Centroid offsets in RA and Dec. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    pa : float
        Position angle [rad].
    dist : float
        Blob radial distance from center. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    ang : float
        Blob azimuthal angle [rad] (East of North in disk frame).
    nxy : int
        Grid pixel count along each dimension.
    dxy : float
        Angular pixel scale. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    version : str
        Evaluation mode (`'chi2'`, `'vis'`, or `'plot'`).

    Returns
    -------
    numpy.ndarray
        2D array of intensity values for the blob.
    """
    xx, yy = get_grid(nxy, dxy)

    # Calculate Cartesian offsets for Blob (East: xdot > 0, North: ydot > 0)
    xdot = dist * np.sin(ang)
    ydot = dist * np.cos(ang)

    if version in ("chi2", "vis"):
        blob_model_jypix = gaussblob_prof(peak, sigma, xx, yy, xdot, ydot, dxy)
        return blob_model_jypix

    elif version == 'plot':
        xx_shifted = xx - dra
        yy_shifted = yy - ddec

        xpa =  xx_shifted * np.cos(pa) - yy_shifted * np.sin(pa)
        ypa =  xx_shifted * np.sin(pa) + yy_shifted * np.cos(pa)

        blob_model_jysr = gaussblob_prof(peak, sigma, xpa, ypa, xdot, ydot, 1)
        return blob_model_jysr

    else:
        msg = f"Invalid version '{version}', must be 'chi2', 'vis', or 'plot'"
        logging.warning(msg)
        raise ValueError(msg)

def twod_simgauss(pars, args, vis_data, version):
    """
    Top-level model handler for Single Gaussian Blob (7 parameters).

    Parameters
    ----------
    pars : array_like (Audit P1 Item 1: log integrated flux + log size)
        [log_flux, log_sigma, dist, ang, posangle, dRA, dDec]
    args : tuple
        `(nxy, dxy)`
    vis_data : tuple
        (u, v, Re, Im, weights)
    version : str
        `'chi2'`, `'vis'`, or `'plot'`
    """
    log_flux, log_sigma, dist, ang, posangle, dRA, dDec = pars
    nxy, dxy = args
    u, v, re, im, w = vis_data

    posangle *= deg
    ang *= deg

    sigma_arcsec = 10.0**log_sigma
    sigma_rad = sigma_arcsec * arcsec

    peak = blob_flux_to_peak(log_flux, sigma_rad)

    if version in ("chi2", "vis"):
        dRA_rad = dRA * arcsec
        dDec_rad = dDec * arcsec
        dist_rad = dist * arcsec
        model_img = twod_simgauss_model(peak, sigma_rad, dRA_rad, dDec_rad, posangle, dist_rad, ang, nxy, dxy, version)

        if version == 'chi2':
            chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower')
            return chi2
        else:
            model_vis = np.array(sampleImage(model_img, dxy, u, v, dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower'), dtype=np.complex256)
            return model_vis
    else:
        model_img = twod_simgauss_model(peak, sigma_arcsec, dRA, dDec, posangle, dist, ang, nxy, dxy, version)
        return model_img

##################################################################################################################################################

def model_prof(pars, args, vis_data, version, fittype):
    """
    Central dispatcher function for model profile evaluation.
    """
    if version not in ("chi2", "vis", "plot"):
        msg = f"Invalid version '{version}', must be 'chi2', 'vis', or 'plot'"
        logging.warning(msg)
        raise ValueError(msg)

    if fittype == 'twod_gaussring':
        prof = twod_gaussring(pars, args, vis_data, version)
    elif fittype == 'twod_gauss1blob':
        prof = twod_gauss1blob(pars, args, vis_data, version)
    elif fittype == 'twod_gauss1blob_2peak':
        prof = twod_gauss1blob_2peak(pars, args, vis_data, version)
    elif fittype == 'twod_gauss1blob_2peak_dp':
        prof = twod_gauss1blob_2peak_dp(pars, args, vis_data, version)
    elif fittype == 'twod_gauss3blob':
        prof = twod_gauss3blob(pars, args, vis_data, version)
    elif fittype in ('simgauss', 'twod_simgauss'):
        prof = twod_simgauss(pars, args, vis_data, version)
    else:
        msg = f"Invalid fittype '{fittype}', please choose a valid fitting model, or add a new one into the code."
        logging.warning(msg)
        raise ValueError(msg)

    return prof

##################################################################################################################################################

def model_addon(fittype):
    """
    Returns parameter metadata (labels, physical units, dimensionality) for a given model.
    Reparameterized according to Audit P1 Item 1: log integrated flux + log size.
    """
    if fittype == 'twod_gaussring':
        label = ["Ring LogFlux", "Ring LogSigma", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec"]
        unit = ["log(Jy)", "log(arcsec)", "arcsec", "degrees", "degrees", "arcsec", "arcsec"]
        ndim = len(label)

    elif fittype == 'twod_gauss1blob':
        label = ["Ring LogFlux", "Ring LogSigma", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec", "B1 LogFlux", "B1 LogSigma", "B1 Dist", "B1 Angle"]
        unit = ["log(Jy)", "log(arcsec)", "arcsec", "degrees", "degrees", "arcsec", "arcsec", "log(Jy)", "log(arcsec)", "arcsec", "degrees"]
        ndim = len(label)
    
    elif fittype == 'twod_gauss1blob_2peak':
        label = ["Ring LogFlux", "Ring LogSigma", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec", "B11 LogFlux", "B11 LogSigma", "B12 LogFlux", "B12 LogSigma", "Dist", "Angle"]
        unit = ["log(Jy)", "log(arcsec)", "arcsec", "degrees", "degrees", "arcsec", "arcsec", "log(Jy)", "log(arcsec)", "log(Jy)", "log(arcsec)", "arcsec", "degrees"]
        ndim = len(label)

    elif fittype == 'twod_gauss1blob_2peak_dp':
        label = ["Ring LogFlux", "Ring LogSigma", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec", "B11 LogFlux", "B11 LogSigma", "Dist 1", "Angle 1", "B12 LogFlux", "B12 LogSigma", "Dist 2", "Angle 2"]
        unit = ["log(Jy)", "log(arcsec)", "arcsec", "degrees", "degrees", "arcsec", "arcsec", "log(Jy)", "log(arcsec)", "arcsec", "degrees", "log(Jy)", "log(arcsec)", "arcsec", "degrees"]
        ndim = len(label)

    elif fittype == 'twod_gauss3blob':
        label = ["Ring LogFlux", "Ring LogSigma", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec", "B1 LogFlux", "B1 LogSigma", "B1 Dist", "B1 Angle", "B2 LogFlux", "B2 LogSigma", "B2 Dist", "B2 Angle", "B3 LogFlux", "B3 LogSigma", "B3 Dist", "B3 Angle"]
        unit = ["log(Jy)", "log(arcsec)", "arcsec", "degrees", "degrees", "arcsec", "arcsec", "log(Jy)", "log(arcsec)", "arcsec", "degrees", "log(Jy)", "log(arcsec)", "arcsec", "degrees", "log(Jy)", "log(arcsec)", "arcsec", "degrees"]
        ndim = len(label)

    elif fittype in ('simgauss', 'twod_simgauss'):
        label = ["Blob LogFlux", "Blob LogSigma", "Dist", "Angle", "PA", "Offset RA", "Offset Dec"]
        unit = ["log(Jy)", "log(arcsec)", "arcsec", "degrees", "degrees", "arcsec", "arcsec"]
        ndim = len(label)

    else:
        msg = f"Invalid fittype '{fittype}', please choose a valid fitting model, or add a new one into the code."
        logging.warning(msg)
        raise ValueError(msg)

    return label, unit, ndim
