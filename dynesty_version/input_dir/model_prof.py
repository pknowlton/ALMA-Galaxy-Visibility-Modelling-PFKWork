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

Execution Modes & Angular Unit Handling (`version` argument):
-------------------------------------------------------------
All spatial parameters—including ring width ($\sigma_{\text{ring}}$), blob widths
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
        Radial standard deviation (Gaussian width / half-width) of the ring.
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
        Gaussian standard deviation (cluster width).
        Expressed in **radians** (for `'chi2'` / `'vis'`) or in **arcseconds** (for `'plot'`).
    xx : numpy.ndarray
        2D meshgrid array of X coordinates, in **radians** or **arcseconds**.
    yy : numpy.ndarray
        2D meshgrid array of Y coordinates, in **radians** or **arcseconds**.
    xoff : float
        X-axis offset of the cluster center from origin, in matching units (**radians** or **arcseconds**).
    yoff : float
        Y-axis offset of the cluster center from origin, in matching units (**radians** or **arcseconds**).
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
    components with distinct peak intensities and standard deviations (widths):
        $I(x, y) = \\left( 10^{\\text{peak}_1} \\cdot \\exp\\left( -\\frac{1}{2} \\left[ \\left(\\frac{x - x_{\\text{off}}}{\\sigma_1}\\right)^2 + \\left(\\frac{y - y_{\\text{off}}}{\\sigma_1}\\right)^2 \\right] \\right) + 10^{\\text{peak}_2} \\cdot \\exp\\left( -\\frac{1}{2} \\left[ \\left(\\frac{x - x_{\\text{off}}}{\\sigma_2}\\right)^2 + \\left(\\frac{y - y_{\\text{off}}}{\\sigma_2}\\right)^2 \\right] \\right) \\right) \\cdot \\text{dxy}^2$

    Parameters
    ----------
    peak1 : float
        Base-10 logarithm of the first component peak surface brightness in Jy/sr.
    sigma1 : float
        Gaussian standard deviation (width) of the first component.
        Expressed in **radians** (for `'chi2'` / `'vis'`) or in **arcseconds** (for `'plot'`).
    peak2 : float
        Base-10 logarithm of the second component peak surface brightness in Jy/sr.
    sigma2 : float
        Gaussian standard deviation (width) of the second component.
        Expressed in **radians** (for `'chi2'` / `'vis'`) or in **arcseconds** (for `'plot'`).
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

# Fixed morphological reference angle for disk rotation
PA_RAD = 15.0 * deg
COS_PA = np.cos(PA_RAD)
SIN_PA = np.sin(PA_RAD)

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
        Gaussian radial width. In **radians** for `version in ('chi2', 'vis')`; in **arcseconds** for `version == 'plot'`.
    rad : float
        Central ring radius. In **radians** for `version in ('chi2', 'vis')`; in **arcseconds** for `version == 'plot'`.
    inc : float
        Disk inclination angle in radians (0 = face-on, $\pi/2$ = edge-on).
    pa : float
        Disk position angle in radians.
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
    # Initialize the image plane grid centered at origin (in radians for chi2/vis, arcsec for plot)
    #image_size = nxy * dxy
    #x = np.linspace(-image_size/2, image_size/2, nxy)
    #y = np.linspace(-image_size/2, image_size/2, nxy)
    #xx, yy = np.meshgrid(x, y)

    x = ((np.arange(nxy) - (nxy-1)/2)-0.5) * dxy
    y = ((np.arange(nxy) - (nxy-1)/2)-0.5) * dxy
    xx, yy = np.meshgrid(x, y)

    if version in ("chi2", "vis"):
        # Deproject x-axis by cos(inclination) to account for disk tilt (all units in radians)
        xinc = xx / np.cos(inc)
        radius_vec = np.hypot(xinc, yy)

        ring_model_jypix = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
        return ring_model_jypix

    elif version == 'plot':
        # Shift coordinates to account for centroid offsets (all units in arcseconds)
        xx_shifted = xx + (dra)
        yy_shifted = yy - (ddec)

        # Rotate coordinates by position angle (PA)
        xpa = xx_shifted * np.cos(pa) + yy_shifted * np.sin(pa)
        ypa = -xx_shifted * np.sin(pa) + yy_shifted * np.cos(pa)

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
    constructs the 2D image model, and evaluates $\chi^2$, model visibilities, or image maps.

    Parameters
    ----------
    pars : array_like
        Model parameter vector:
        [peak (log Jy/sr), sigma (arcsec), ring_rad (arcsec), inclination (deg),
         posangle (deg), dRA (arcsec), dDec (arcsec)]
        *Note*: Spatial parameters (`sigma`, `ring_rad`, `dRA`, `dDec`) are passed in **arcseconds**.
        When `version in ('chi2', 'vis')`, they are automatically converted to **radians**.
        When `version == 'plot'`, they remain in **arcseconds**.
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
        - If `version == 'chi2'`: Scalar goodness-of-fit $\chi^2$ value.
        - If `version == 'vis'`: 1D complex numpy array of sampled model visibilities.
        - If `version == 'plot'`: 2D real numpy array of model image brightness.
    """

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec = pars
    nxy, dxy = args
    u, v, re, im, w = vis_data

    inclination *= deg
    posangle *= deg

    # Convert spatial parameters from arcseconds to radians for Galario visibility domain
    if version in ("chi2", "vis"):
        sigma *= arcsec
        ring_rad *= arcsec
        dRA *= arcsec
        dDec *= arcsec

    model_img = twod_gaussring_model(peak, sigma, ring_rad, inclination, posangle, dRA, dDec, nxy, dxy, version) 

    if version == 'chi2':

        chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA, dDec=dDec, PA=posangle, origin='lower')

        return chi2

    elif version == 'vis':

        model_vis = np.array(sampleImage(model_img, dxy, u, v, dRA=dRA, dDec=dDec, PA=posangle, origin='lower'), dtype=np.complex256)

        return model_vis

    else:

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
        Ring radial width and central radius. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    inc, pa : float
        Disk inclination and position angle [rad].
    dra, ddec : float
        Centroid offsets in RA and Dec. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    peak_b1 : float
        Blob 1 peak brightness [log10(Jy/sr)].
    sigma_b1, dist_b1 : float
        Blob 1 Gaussian width and radial distance from ring center. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    ang_b1 : float
        Blob 1 azimuthal angle [rad].
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
    # Initialize the image plane
    #image_size = nxy * dxy
    #x = np.linspace(-image_size/2, image_size/2, nxy)
    #y = np.linspace(-image_size/2, image_size/2, nxy)
    #xx, yy = np.meshgrid(x, y)

    x = ((np.arange(nxy) - (nxy-1)/2)-0.5) * dxy
    y = ((np.arange(nxy) - (nxy-1)/2)-0.5) * dxy
    xx, yy = np.meshgrid(x, y)

    if version in ("chi2", "vis"):

        # Apply structural position angle rotation (all spatial units in radians)
        xpa_rg = xx*COS_PA + yy*SIN_PA
        ypa_rg = -xx*SIN_PA + yy*COS_PA
        
        xinc = xpa_rg/np.cos(inc)
        radius_vec = np.hypot(xinc, ypa_rg)

        # Calculate Cartesian offsets for Blob 1 (in radians)
        xdot_b1 = dist_b1 * np.sin(ang_b1)
        ydot_b1 = dist_b1 * np.cos(ang_b1)

        ring_model_jypix = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
        blob1_model_jypix = gaussblob_prof(peak_b1, sigma_b1, xx, yy, xdot_b1, ydot_b1, dxy)

        return ring_model_jypix +  blob1_model_jypix

    elif version=='plot':

        # Shift and rotate coordinates (all spatial units in arcseconds)
        xx_shifted = xx + (dra)
        yy_shifted = yy - (ddec)

        xpa = xx_shifted*np.cos(pa) + yy_shifted*np.sin(pa)
        ypa = -xx_shifted*np.sin(pa) + yy_shifted*np.cos(pa)

        xpa_rg = xpa*COS_PA + ypa*SIN_PA
        ypa_rg = -xpa*SIN_PA + ypa*COS_PA

        xinc = xpa_rg/np.cos(inc)
        radius_vec = np.hypot(xinc, ypa_rg)

        xdot_b1 = dist_b1 * np.sin(ang_b1)
        ydot_b1 = dist_b1 * np.cos(ang_b1)

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

    Converts spatial parameter units (arcseconds to radians for fitting modes) and
    interfaces with Galario to compute $\chi^2$, visibilities, or image arrays.

    Parameters
    ----------
    pars : array_like
        Parameter array: [peak, sigma, ring_rad, inclination, posangle, dRA, dDec,
                          peak_b1, sigma_b1, dist_b1, ang_b1].
        *Note*: `sigma`, `ring_rad`, `dRA`, `dDec`, `sigma_b1`, and `dist_b1` are passed in **arcseconds**.
        They are automatically converted to **radians** when `version in ('chi2', 'vis')`,
        and kept in **arcseconds** when `version == 'plot'`.
    args : tuple
        Image grid parameters: `(nxy, dxy)` (where `dxy` is in radians for chi2/vis, arcsec for plot).
    vis_data : tuple
        Visibility data: (u, v, Re, Im, weights).
    version : str
        Evaluation mode: `'chi2'`, `'vis'`, or `'plot'`.

    Returns
    -------
    float or numpy.ndarray
        $\chi^2$ value, synthetic visibilities array, or 2D image matrix.
    """

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b1, sigma_b1, dist_b1, ang_b1 = pars
    nxy, dxy = args
    u, v, re, im, w = vis_data

    inclination *= deg
    posangle *= deg
    ang_b1 *= deg

    # Convert spatial parameters from arcseconds to radians for Galario visibility domain
    if version in ("chi2", "vis"):

        sigma *= arcsec
        ring_rad *= arcsec
        dRA *= arcsec
        dDec *= arcsec
        sigma_b1 *= arcsec
        dist_b1 *= arcsec

    model_img = twod_gauss1blob_model(peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b1, sigma_b1, dist_b1, ang_b1, nxy, dxy, version) 

    if version == 'chi2':

        chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA, dDec=dDec, PA=posangle, origin='lower')

        return chi2

    elif version == 'vis':

        model_vis = np.array(sampleImage(model_img, dxy, u, v, dRA=dRA, dDec=dDec, PA=posangle, origin='lower'), dtype=np.complex256)

        return model_vis

    else:

        return model_img


#########################
### 2D Gaussian Ring + 1 Gaussian Blob (2 Peaks)
#########################

def twod_gauss1blob_2peak_model(peak, sigma, rad, inc, pa, dra, ddec, peak_b11, sigma_b11, peak_b12, sigma_b12, dist_b1, ang_b1, nxy, dxy, version):
    """
    Generates a 2D image matrix of an inclined Gaussian ring plus 1 composite 2-peak Gaussian cluster.

    Parameters
    ----------
    peak : float
        Ring peak brightness [log10(Jy/sr)].
    sigma, rad : float
        Ring radial width and central radius. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    inc, pa : float
        Disk inclination and position angle [rad].
    dra, ddec : float
        Centroid offsets in RA and Dec. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    peak_b11 : float
        Blob 1 Component 1 peak brightness [log10(Jy/sr)].
    sigma_b11 : float
        Blob 1 Component 1 Gaussian width. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    peak_b12 : float
        Blob 1 Component 2 peak brightness [log10(Jy/sr)].
    sigma_b12 : float
        Blob 1 Component 2 Gaussian width. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    dist_b1 : float
        Blob 1 radial distance from ring center. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    ang_b1 : float
        Blob 1 azimuthal angle [rad].
    nxy : int
        Grid pixel count along each dimension.
    dxy : float
        Angular pixel scale. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    version : str
        Evaluation mode (`'chi2'`, `'vis'`, or `'plot'`).

    Returns
    -------
    numpy.ndarray
        2D array representing combined ring and 2-peak blob intensity.
    """
    # Initialize the image plane
    #image_size = nxy * dxy
    #x = np.linspace(-image_size/2, image_size/2, nxy)
    #y = np.linspace(-image_size/2, image_size/2, nxy)
    #xx, yy = np.meshgrid(x, y)

    x = ((np.arange(nxy) - (nxy-1)/2)-0.5) * dxy
    y = ((np.arange(nxy) - (nxy-1)/2)-0.5) * dxy
    xx, yy = np.meshgrid(x, y)

    if version in ("chi2", "vis"):

        # Apply structural position angle rotation (all spatial units in radians)
        xpa_rg = xx*COS_PA + yy*SIN_PA
        ypa_rg = -xx*SIN_PA + yy*COS_PA
        
        xinc = xpa_rg/np.cos(inc)
        radius_vec = np.hypot(xinc, ypa_rg)

        # Calculate Cartesian offsets for Blob 1 (in radians)
        xdot_b1 = dist_b1 * np.sin(ang_b1)
        ydot_b1 = dist_b1 * np.cos(ang_b1)

        ring_model_jypix = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
        blob1_model_jypix = gaussblob_2peak_prof(peak_b11, sigma_b11, peak_b12, sigma_b12, xx, yy, xdot_b1, ydot_b1, dxy)

        return ring_model_jypix + blob1_model_jypix

    elif version == 'plot':

        # Shift and rotate coordinates (all spatial units in arcseconds)
        xx_shifted = xx + (dra)
        yy_shifted = yy - (ddec)

        xpa = xx_shifted*np.cos(pa) + yy_shifted*np.sin(pa)
        ypa = -xx_shifted*np.sin(pa) + yy_shifted*np.cos(pa)

        xpa_rg = xpa*COS_PA + ypa*SIN_PA
        ypa_rg = -xpa*SIN_PA + ypa*COS_PA

        xinc = xpa_rg/np.cos(inc)
        radius_vec = np.hypot(xinc, ypa_rg)

        xdot_b1 = dist_b1 * np.sin(ang_b1)
        ydot_b1 = dist_b1 * np.cos(ang_b1)

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

    Converts spatial parameter units (arcseconds to radians for fitting modes) and
    interfaces with Galario to compute $\chi^2$, visibilities, or image arrays.

    Parameters
    ----------
    pars : array_like
        Parameter array: [peak, sigma, ring_rad, inclination, posangle, dRA, dDec,
                          peak_b11, sigma_b11, peak_b12, sigma_b12, dist_b1, ang_b1].
        *Note*: `sigma`, `ring_rad`, `dRA`, `dDec`, `sigma_b11`, `sigma_b12`, and `dist_b1` are passed in **arcseconds**.
        They are automatically converted to **radians** when `version in ('chi2', 'vis')`,
        and kept in **arcseconds** when `version == 'plot'`.
    args : tuple
        Image grid parameters: `(nxy, dxy)` (where `dxy` is in radians for chi2/vis, arcsec for plot).
    vis_data : tuple
        Visibility data: (u, v, Re, Im, weights).
    version : str
        Evaluation mode: `'chi2'`, `'vis'`, or `'plot'`.

    Returns
    -------
    float or numpy.ndarray
        $\chi^2$ value, synthetic visibilities array, or 2D image matrix.
    """

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b11, sigma_b11, peak_b12, sigma_b12, dist_b1, ang_b1 = pars
    nxy, dxy = args
    u, v, re, im, w = vis_data

    inclination *= deg
    posangle *= deg
    ang_b1 *= deg

    # Convert spatial parameters from arcseconds to radians for Galario visibility domain
    if version in ("chi2", "vis"):

        sigma *= arcsec
        ring_rad *= arcsec
        dRA *= arcsec
        dDec *= arcsec
        sigma_b11 *= arcsec
        sigma_b12 *= arcsec
        dist_b1 *= arcsec

    model_img = twod_gauss1blob_2peak_model(peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b11, sigma_b11, peak_b12, sigma_b12, dist_b1, ang_b1, nxy, dxy, version) 

    if version == 'chi2':

        chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA, dDec=dDec, PA=posangle, origin='lower')

        return chi2

    elif version == 'vis':

        model_vis = np.array(sampleImage(model_img, dxy, u, v, dRA=dRA, dDec=dDec, PA=posangle, origin='lower'), dtype=np.complex256)

        return model_vis

    else:

        return model_img

#########################
### 2D Gaussian Ring + 1 Gaussian Blob (2 Peaks, Double Pendulum)
#########################


def twod_gauss1blob_2peak_dp_model(peak, sigma, rad, inc, pa, dra, ddec, peak_b11, sigma_b11, dist_b11, ang_b11, peak_b12, sigma_b12, dist_b12, ang_b12, nxy, dxy, version):
    """
    Constructs a 2D intensity grid for a tilted Gaussian ring with a 2-peak "Double Pendulum" clump.

    The model consists of a smooth elliptical Gaussian ring plus two Gaussian emission peaks
    forming a single complex clump. Peak 1 is positioned at polar coordinates (dist_b11, ang_b11)
    relative to the galaxy center. Peak 2 is positioned at relative polar coordinates (dist_b12, ang_b12)
    with respect to Peak 1 (with separation dist_b12 <= 2 arcsec).

    Parameters
    ----------
    peak : float
        Ring peak brightness [log10(Jy/sr)].
    sigma, rad : float
        Ring radial width and central radius. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    inc, pa : float
        Disk inclination and position angle [rad].
    dra, ddec : float
        Centroid offsets in RA and Dec. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    peak_b11 : float
        Blob Component 1 peak brightness [log10(Jy/sr)].
    sigma_b11 : float
        Blob Component 1 Gaussian standard deviation (width). In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    dist_b11 : float
        Blob Component 1 radial distance from ring center. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    ang_b11 : float
        Blob Component 1 azimuthal position angle [rad].
    peak_b12 : float
        Blob Component 2 peak brightness [log10(Jy/sr)].
    sigma_b12 : float
        Blob Component 2 Gaussian standard deviation (width). In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    dist_b12 : float
        Blob Component 2 separation distance from Peak 1 (<= 2 arcsec). In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    ang_b12 : float
        Blob Component 2 position angle relative to Peak 1 [rad].
    nxy : int
        Grid pixel count along each dimension.
    dxy : float
        Angular pixel scale. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    version : str
        Evaluation mode (`'chi2'`, `'vis'`, or `'plot'`).

    Returns
    -------
    numpy.ndarray
        2D array representing combined ring and 2-peak double pendulum blob intensity.
    """
    # Initialize the image plane
    #image_size = nxy * dxy
    #x = np.linspace(-image_size/2, image_size/2, nxy)
    #y = np.linspace(-image_size/2, image_size/2, nxy)
    #xx, yy = np.meshgrid(x, y)

    x = ((np.arange(nxy) - (nxy-1)/2)-0.5) * dxy
    y = ((np.arange(nxy) - (nxy-1)/2)-0.5) * dxy
    xx, yy = np.meshgrid(x, y)

    if version in ("chi2", "vis"):

        # Apply structural position angle rotation (all spatial units in radians)
        xpa_rg = xx*COS_PA + yy*SIN_PA
        ypa_rg = -xx*SIN_PA + yy*COS_PA
        
        xinc = xpa_rg/np.cos(inc)
        radius_vec = np.hypot(xinc, ypa_rg)

        # Calculate Cartesian offsets for Blob 1 Peak 1 (in radians)
        xdot_b11 = dist_b11 * np.sin(ang_b11)
        ydot_b11 = dist_b11 * np.cos(ang_b11)

        # Calculate extra Cartesian offsets for Blob 1 Peak 2 (in radians)
        xdot_b12 = dist_b12 * np.sin(ang_b12)
        ydot_b12 = dist_b12 * np.cos(ang_b12)
        xdot_bdp = xdot_b11 + xdot_b12
        ydot_bdp = ydot_b11 + ydot_b12

        ring_model_jypix = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
        blob11_model_jypix = gaussblob_prof(peak_b11, sigma_b11, xx, yy, xdot_b11, ydot_b11, dxy)
        blob12_model_jypix = gaussblob_prof(peak_b12, sigma_b12, xx, yy, xdot_bdp, ydot_bdp, dxy)

        return ring_model_jypix + blob11_model_jypix + blob12_model_jypix

    elif version == 'plot':

        # Shift and rotate coordinates (all spatial units in arcseconds)
        xx_shifted = xx + (dra)
        yy_shifted = yy - (ddec)

        xpa = xx_shifted*np.cos(pa) + yy_shifted*np.sin(pa)
        ypa = -xx_shifted*np.sin(pa) + yy_shifted*np.cos(pa)

        xpa_rg = xpa*COS_PA + ypa*SIN_PA
        ypa_rg = -xpa*SIN_PA + ypa*COS_PA

        xinc = xpa_rg/np.cos(inc)
        radius_vec = np.hypot(xinc, ypa_rg)

        # Calculate Cartesian offsets for Blob 1 Peak 1 (in arcseconds)
        xdot_b11 = dist_b11 * np.sin(ang_b11)
        ydot_b11 = dist_b11 * np.cos(ang_b11)

        # Calculate extra Cartesian offsets for Blob 1 Peak 2 (in arcseconds)
        xdot_b12 = dist_b12 * np.sin(ang_b12)
        ydot_b12 = dist_b12 * np.cos(ang_b12)
        xdot_bdp = xdot_b11 + xdot_b12
        ydot_bdp = ydot_b11 + ydot_b12

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

    Converts spatial parameter units (arcseconds to radians for fitting modes) and
    interfaces with Galario to compute $\chi^2$, visibilities, or image arrays.

    Parameters
    ----------
    pars : array_like
        Parameter array: [peak, sigma, ring_rad, inclination, posangle, dRA, dDec,
                          peak_b11, sigma_b11, dist_b11, ang_b11,
                          peak_b12, sigma_b12, dist_b12, ang_b12].
        *Note*: `sigma`, `ring_rad`, `dRA`, `dDec`, `sigma_b11`, `dist_b11`, `sigma_b12`, and `dist_b12`
        are passed in **arcseconds**. They are automatically converted to **radians**
        when `version in ('chi2', 'vis')`, and kept in **arcseconds** when `version == 'plot'`.
    args : tuple
        Image grid parameters: `(nxy, dxy)` (where `dxy` is in radians for chi2/vis, arcsec for plot).
    vis_data : tuple
        Visibility data: (u, v, Re, Im, weights).
    version : str
        Evaluation mode: `'chi2'`, `'vis'`, or `'plot'`.

    Returns
    -------
    float or numpy.ndarray
        $\chi^2$ value, synthetic visibilities array, or 2D image matrix.
    """

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b11, sigma_b11, dist_b11, ang_b11, peak_b12, sigma_b12, dist_b12, ang_b12 = pars
    nxy, dxy = args
    u, v, re, im, w = vis_data

    inclination *= deg
    posangle *= deg
    ang_b11 *= deg
    ang_b12 *= deg

    # Convert spatial parameters from arcseconds to radians for Galario visibility domain
    if version in ("chi2", "vis"):

        sigma *= arcsec
        ring_rad *= arcsec
        dRA *= arcsec
        dDec *= arcsec
        sigma_b11 *= arcsec
        sigma_b12 *= arcsec
        dist_b11 *= arcsec
        dist_b12 *= arcsec

    model_img = twod_gauss1blob_2peak_dp_model(peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b11, sigma_b11, dist_b11, ang_b11, peak_b12, sigma_b12, dist_b12, ang_b12, nxy, dxy, version) 

    if version == 'chi2':

        chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA, dDec=dDec, PA=posangle, origin='lower')

        return chi2

    elif version == 'vis':

        model_vis = np.array(sampleImage(model_img, dxy, u, v, dRA=dRA, dDec=dDec, PA=posangle, origin='lower'), dtype=np.complex256)

        return model_vis

    else:

        return model_img


#########################
### 2D Gaussian Ring + 3 Gaussian Blobs
#########################

def twod_gauss3blob_model(peak, sigma, rad, inc, pa, dra, ddec, peak_b1, sigma_b1, dist_b1, ang_b1, peak_b2, sigma_b2, dist_b2, ang_b2, peak_b3, sigma_b3, dist_b3, ang_b3, nxy, dxy, version):
    """
    Generates a 2D image matrix of an inclined Gaussian ring plus 3 distinct Gaussian clusters.

    Parameters
    ----------
    peak : float
        Ring peak brightness [log10(Jy/sr)].
    sigma, rad : float
        Ring radial width and central radius. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    inc, pa : float
        Disk inclination and position angle [rad].
    dra, ddec : float
        Centroid offsets in RA and Dec. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    peak_b1..3 : float
        Peak brightness of Blobs 1, 2, and 3 [log10(Jy/sr)].
    sigma_b1..3, dist_b1..3 : float
        Widths and radial distances for Blobs 1, 2, and 3.
        In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    ang_b1..3 : float
        Azimuthal position angles for Blobs 1, 2, and 3 [rad].
    nxy : int
        Grid dimension in pixels.
    dxy : float
        Pixel scale. In **radians** for `'chi2'`/`'vis'`; in **arcseconds** for `'plot'`.
    version : str
        Evaluation mode (`'chi2'`, `'vis'`, or `'plot'`).

    Returns
    -------
    numpy.ndarray
        2D array representing combined ring and 3-blob intensity.
    """
    # Initialize the image plane
    #image_size = nxy * dxy
    #x = np.linspace(-image_size/2, image_size/2, nxy)
    #y = np.linspace(-image_size/2, image_size/2, nxy)
    #xx, yy = np.meshgrid(x, y)

    x = ((np.arange(nxy) - (nxy-1)/2)-0.5) * dxy
    y = ((np.arange(nxy) - (nxy-1)/2)-0.5) * dxy
    xx, yy = np.meshgrid(x, y)

    if version in ("chi2", "vis"):

        # Structural position angle rotation (all spatial units in radians)
        xpa_rg = xx*COS_PA + yy*SIN_PA
        ypa_rg = -xx*SIN_PA + yy*COS_PA
        
        xinc = xpa_rg/np.cos(inc)
        radius_vec = np.hypot(xinc, ypa_rg)

        xdot_b1 = dist_b1 * np.sin(ang_b1)
        ydot_b1 = dist_b1 * np.cos(ang_b1)

        xdot_b2 = dist_b2 * np.sin(ang_b2)
        ydot_b2 = dist_b2 * np.cos(ang_b2)

        xdot_b3 = dist_b3 * np.sin(ang_b3)
        ydot_b3 = dist_b3 * np.cos(ang_b3)

        ring_model_jypix = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
        blob1_model_jypix = gaussblob_prof(peak_b1, sigma_b1, xx, yy, xdot_b1, ydot_b1, dxy)
        blob2_model_jypix = gaussblob_prof(peak_b2, sigma_b2, xx, yy, xdot_b2, ydot_b2, dxy)
        blob3_model_jypix = gaussblob_prof(peak_b3, sigma_b3, xx, yy, xdot_b3, ydot_b3, dxy)

        return ring_model_jypix + blob1_model_jypix + blob2_model_jypix + blob3_model_jypix

    elif version=='plot':

        # Shift and rotate coordinates (all spatial units in arcseconds)
        xx_shifted = xx + (dra)
        yy_shifted = yy - (ddec)

        xpa = xx_shifted*np.cos(pa) + yy_shifted*np.sin(pa)
        ypa = -xx_shifted*np.sin(pa) + yy_shifted*np.cos(pa)

        xpa_rg = xpa*COS_PA + ypa*SIN_PA
        ypa_rg = -xpa*SIN_PA + ypa*COS_PA

        xinc = xpa_rg/np.cos(inc)
        radius_vec = np.hypot(xinc, ypa_rg)

        xdot_b1 = dist_b1 * np.sin(ang_b1)
        ydot_b1 = dist_b1 * np.cos(ang_b1)

        xdot_b2 = dist_b2 * np.sin(ang_b2)
        ydot_b2 = dist_b2 * np.cos(ang_b2)

        xdot_b3 = dist_b3 * np.sin(ang_b3)
        ydot_b3 = dist_b3 * np.cos(ang_b3)

        ring_model_jysr = gaussring_prof(peak, sigma, rad, radius_vec, 1)
        blob1_model_jysr = gaussblob_prof(peak_b1, sigma_b1, xpa, ypa, xdot_b1, ydot_b1, 1)
        blob2_model_jysr = gaussblob_prof(peak_b2, sigma_b2, xpa, ypa, xdot_b2, ydot_b2, 1)
        blob3_model_jysr = gaussblob_prof(peak_b3, sigma_b3, xpa, ypa, xdot_b3, ydot_b3, 1)

        return ring_model_jysr + blob1_model_jysr + blob2_model_jysr + blob3_model_jysr
        #return blob1_model_jysr + blob2_model_jysr + blob3_model_jysr

    else:

        msg = f"Invalid version '{version}', must be 'chi2', 'vis', or 'plot'"
        logging.warning(msg)
        raise ValueError(msg)

def twod_gauss3blob(pars, args, vis_data, version):
    """
    Top-level model handler for 2D Gaussian Ring + 3 Gaussian Blobs (19 parameters).

    Converts spatial parameter units (arcseconds to radians for fitting modes) and
    interfaces with Galario to compute $\chi^2$, visibilities, or image arrays.

    Parameters
    ----------
    pars : array_like
        19-element parameter array for the ring and three blobs:
        [peak, sigma, ring_rad, inc, pa, dra, ddec,
         peak_b1, sigma_b1, dist_b1, ang_b1,
         peak_b2, sigma_b2, dist_b2, ang_b2,
         peak_b3, sigma_b3, dist_b3, ang_b3].
        *Note*: All width ($\sigma$), radius ($R$), offset ($\Delta\alpha, \Delta\delta$),
        and distance ($d$) parameters are supplied in **arcseconds**.
        They are automatically converted to **radians** when `version in ('chi2', 'vis')`,
        and kept in **arcseconds** when `version == 'plot'`.
    args : tuple
        Image grid parameters: `(nxy, dxy)` (where `dxy` is in radians for chi2/vis, arcsec for plot).
    vis_data : tuple
        Visibility data: (u, v, Re, Im, weights).
    version : str
        Evaluation mode: `'chi2'`, `'vis'`, or `'plot'`.

    Returns
    -------
    float or numpy.ndarray
        $\chi^2$ value, synthetic visibilities array, or 2D image matrix.
    """

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b1, sigma_b1, dist_b1, ang_b1, peak_b2, sigma_b2, dist_b2, ang_b2, peak_b3, sigma_b3, dist_b3, ang_b3 = pars
    nxy, dxy = args
    u, v, re, im, w = vis_data

    inclination *= deg
    posangle *= deg
    ang_b1 *= deg
    ang_b2 *= deg
    ang_b3 *= deg

    # Convert spatial parameters from arcseconds to radians for Galario visibility domain
    if version in ("chi2", "vis"):

        sigma *= arcsec
        ring_rad *= arcsec
        dRA *= arcsec
        dDec *= arcsec
        sigma_b1 *= arcsec
        dist_b1 *= arcsec
        sigma_b2 *= arcsec
        dist_b2 *= arcsec
        sigma_b3 *= arcsec
        dist_b3 *= arcsec

    model_img = twod_gauss3blob_model(peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b1, sigma_b1, dist_b1, ang_b1, peak_b2, sigma_b2, dist_b2, ang_b2, peak_b3, sigma_b3, dist_b3, ang_b3, nxy, dxy, version) 

    if version == 'chi2':

        chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA, dDec=dDec, PA=posangle, origin='lower')

        return chi2

    elif version == 'vis':

        model_vis = np.array(sampleImage(model_img, dxy, u, v, dRA=dRA, dDec=dDec, PA=posangle, origin='lower'), dtype=np.complex256)

        return model_vis

    else:

        return model_img


##################################################################################################################################################


def model_prof(pars, args, vis_data, version, fittype):
    """
    Central dispatcher function for model profile evaluation.

    Routes parameter values and observational visibility data to the specific
    geometric model function corresponding to `fittype`.

    Parameters
    ----------
    pars : array_like
        1D parameter vector to evaluate (spatial parameters in arcseconds, angles in degrees).
    args : tuple
        Grid properties `(nxy, dxy)` (where `dxy` is in radians for chi2/vis, arcsec for plot).
    vis_data : tuple
        Observational visibility data tuple: (u, v, Re, Im, weights).
    version : str
        Execution mode:
        - `'chi2'`: Computes scalar goodness-of-fit $\chi^2$ via Galario.
        - `'vis'`: Computes complex synthetic visibilities via Galario.
        - `'plot'`: Generates 2D model sky brightness map in $\text{Jy}/\text{sr}$.
    fittype : str
        Model identifier: `'twod_gaussring'`, `'twod_gauss1blob'`, `'twod_gauss1blob_2peak'`, `'twod_gauss1blob_2peak_dp'`, or `'twod_gauss3blob'`.

    Returns
    -------
    float or numpy.ndarray
        - $\chi^2$ value (if `version == 'chi2'`).
        - Model visibilities (if `version == 'vis'`).
        - Model sky brightness 2D image (if `version == 'plot'`).

    Raises
    ------
    ValueError
        If `version` or `fittype` is not recognized.
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

    else:
        msg = f"Invalid fittype '{fittype}', please choose a valid fitting model, or add a new one into the code."
        logging.warning(msg)
        raise ValueError(msg)

    return prof

##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################


def model_addon(fittype):
    """
    Returns parameter metadata (labels, physical units, dimensionality) for a given model.

    Parameters
    ----------
    fittype : str
        Model identifier: `'twod_gaussring'`, `'twod_gauss1blob'`, `'twod_gauss1blob_2peak'`, `'twod_gauss1blob_2peak_dp'`, or `'twod_gauss3blob'`.

    Returns
    -------
    label : list of str
        Human-readable parameter names (e.g. "Peak", "Width", "Ring Rad", "Inc", "PA").
    unit : list of str
        Physical units corresponding to each parameter (e.g. "log(Jy/sr)", "arcsec", "degrees").
    ndim : int
        Total number of free parameters for the specified model.

    Raises
    ------
    ValueError
        If `fittype` is not recognized.
    """

    if fittype == 'twod_gaussring':
        label = ["Peak", "Width", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec"]
        unit = ["log(Jy/sr)", "arcsec", "arcsec", "degrees", "degrees", "arcsec", "arcsec"]
        ndim = len(label)

    elif fittype == 'twod_gauss1blob':
        label = ["Peak", "Width", "Offset", "Inc", "PA", "Offset RA", "Offset Dec", "B. Peak", "B. Width", "Dist", "Angle"]
        unit = ["log(Jy/sr)", "arcsec", "arcsec", "degrees", "degrees", "arcsec", "arcsec", "log(Jy/sr)", "arcsec", "arcsec", "degrees"]
        ndim = len(label)
    
    elif fittype == 'twod_gauss1blob_2peak':
        label = ["Peak", "Width", "Offset", "Inc", "PA", "Offset RA", "Offset Dec", "B. Peak 1", "B. Width 1", "B. Peak 2", "B. Width 2", "Dist", "Angle"]
        unit = ["log(Jy/sr)", "arcsec", "arcsec", "degrees", "degrees", "arcsec", "arcsec", "log(Jy/sr)", "arcsec", "log(Jy/sr)", "arcsec", "arcsec", "degrees"]
        ndim = len(label)

    elif fittype == 'twod_gauss1blob_2peak_dp':
        label = ["Peak", "Width", "Offset", "Inc", "PA", "Offset RA", "Offset Dec", "B. Peak 1", "B. Width 1", "Dist 1", "Angle 1", "B. Peak 2", "B. Width 2", "Dist 2", "Angle 2"]
        unit = ["log(Jy/sr)", "arcsec", "arcsec", "degrees", "degrees", "arcsec", "arcsec", "log(Jy/sr)", "arcsec", "arcsec", "degrees", "log(Jy/sr)", "arcsec", "arcsec", "degrees"]
        ndim = len(label)

    elif fittype == 'twod_gauss3blob':
        label = ["Peak", "Width", "Offset", "Inc", "PA", "Offset RA", "Offset Dec", "B1. Peak", "B1. Width", "B1. Dist", "B1. Angle", "B2. Peak", "B2. Width", "B2. Dist", "B2. Angle", "B3. Peak", "B3. Width", "B3. Dist", "B3. Angle"]
        unit = ["log(Jy/sr)", "arcsec", "arcsec", "degrees", "degrees", "arcsec", "arcsec", "log(Jy/sr)", "arcsec", "arcsec", "degrees", "log(Jy/sr)", "arcsec", "arcsec", "degrees", "log(Jy/sr)", "arcsec", "arcsec", "degrees"]
        ndim = len(label)

    else:
        msg = f"Invalid fittype '{fittype}', please choose a valid fitting model, or add a new one into the code."
        logging.warning(msg)
        raise ValueError(msg)

    return label, unit, ndim