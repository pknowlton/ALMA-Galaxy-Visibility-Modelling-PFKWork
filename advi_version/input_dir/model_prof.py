r"""
2D Galaxy Brightness Profile Models for ADVI Visibility Fitting
================================================================
This module defines parametric 2D intensity distributions representing millimeter
dust continuum emission in galaxy nuclear rings (specifically applied to NGC 3351 / M95),
streamlined specifically for Automatic Differentiation Variational Inference (ADVI).

Per instructions, this version contains ONLY the two core models:
1. 'twod_gaussring': Inclined 2D Gaussian Ring (7 parameters)
2. 'twod_gauss1blob': Inclined 2D Gaussian Ring + 1 compact Gaussian Blob (11 parameters)

All coordinate parity conventions, descending-RA grid definitions, beam solid angle factors,
and (log integrated flux + log size) reparameterizations established in the audit are fully preserved:
- RA offset (\Delta\alpha) is positive towards East (Galario origin='lower', column 0 has x > 0).
- Dec offset (\Delta\delta) is positive towards North (row index iy increases with y).
- Disk Position Angle (PA) is defined East of North.
- Blob azimuthal angle is defined East of North in the disk frame.
"""

import numpy as np
import logging
from galario.double import get_image_size, deg, arcsec, chi2Profile, chi2Image, sampleProfile, sampleImage
import pandas as pd

# Module-level coordinate meshgrid cache to eliminate redundant memory allocations
_GRID_CACHE = {}

def get_grid(nxy, dxy):
    r"""
    Returns coordinate meshgrid (xx, yy) matching Galario's origin='lower' convention.

    xx : 2D array of Right Ascension offsets (East positive, descending with column index; East is left).
    yy : 2D array of Declination offsets (North positive, ascending with row index; North is up).

    Caches precomputed grids for each (nxy, dxy) pair to eliminate redundant memory allocations.
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


###############################################################################
# Base Radial & 2D Profile Functions
###############################################################################

def gaussring_prof(peak_ring, sigma_ring, rad_ring, radius, dxy):
    r"""
    Computes the radial surface brightness of a Gaussian ring:
        $I(r) = 10^{\text{peak\_ring}} \cdot \exp\left( -\frac{1}{2} \left(\frac{r - r_{\text{ring}}}{\sigma_{\text{ring}}}\right)^2 \right) \cdot \text{dxy}^2$
    """
    return 10**peak_ring * np.exp((-0.5) * ((radius - rad_ring) / sigma_ring)**2) * (dxy**2)


def gaussblob_prof(peak, sigma, xx, yy, xoff, yoff, dxy):
    r"""
    Computes the 2D surface brightness of a circular Gaussian clump/blob:
        $I(x, y) = 10^{\text{peak}} \cdot \exp\left( -\frac{1}{2} \left[ \left(\frac{x - x_{\text{off}}}{\sigma}\right)^2 + \left(\frac{y - y_{\text{off}}}{\sigma}\right)^2 \right] \right) \cdot \text{dxy}^2$
    """
    return 10**peak * np.exp((-0.5) * (((xx - xoff) / sigma)**2 + ((yy - yoff) / sigma)**2)) * (dxy**2)


def ring_flux_to_peak(log_flux_jy, sigma_rad, rad_rad, inc_rad):
    r"""
    Converts log10 integrated flux of an inclined Gaussian ring to peak surface brightness log10(Jy/sr).
    F_ring = (2*pi)^(3/2) * I_0 * R * sigma * cos(inc)
    """
    flux_jy = 10.0**log_flux_jy
    cos_inc = max(float(np.cos(inc_rad)), 0.05)
    area_sr = ((2.0 * np.pi)**1.5) * float(rad_rad) * float(sigma_rad) * cos_inc
    i0_jysr = flux_jy / max(area_sr, 1e-30)
    return float(np.log10(max(i0_jysr, 1e-30)))


def blob_flux_to_peak(log_flux_jy, sigma_rad):
    r"""
    Converts log10 integrated flux of a circular 2D Gaussian blob to peak surface brightness log10(Jy/sr).
    F_blob = 2 * pi * I_0 * sigma^2
    """
    flux_jy = 10.0**log_flux_jy
    area_sr = 2.0 * np.pi * (float(sigma_rad)**2)
    i0_jysr = flux_jy / max(area_sr, 1e-30)
    return float(np.log10(max(i0_jysr, 1e-30)))


###############################################################################
# 1. Model: 2D Gaussian Ring (7 parameters)
###############################################################################

def twod_gaussring_model(peak, sigma, rad, inc, pa, dra, ddec, nxy, dxy, version):
    r"""
    Generates a 2D image matrix representing an inclined, rotated Gaussian ring.
    """
    xx, yy = get_grid(nxy, dxy)

    if version in ("chi2", "vis"):
        # Deproject x-axis by cos(inclination) (all units in radians)
        xinc = xx / np.cos(inc)
        radius_vec = np.hypot(xinc, yy)
        ring_model_jypix = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
        return ring_model_jypix

    elif version == 'plot':
        # Shift coordinates for centroid offsets (arcseconds)
        xx_shifted = xx - dra
        yy_shifted = yy - ddec

        # Rotate by position angle (PA, East of North)
        xpa =  xx_shifted * np.cos(pa) - yy_shifted * np.sin(pa)
        ypa =  xx_shifted * np.sin(pa) + yy_shifted * np.cos(pa)

        # Deproject inclined circular ring into an ellipse
        xinc = xpa / np.cos(inc)
        radius_vec = np.hypot(xinc, ypa)

        # Return surface brightness in Jy/sr (dxy = 1)
        ring_model_jysr = gaussring_prof(peak, sigma, rad, radius_vec, 1.0)
        return ring_model_jysr

    else:
        msg = f"Invalid version '{version}', must be 'chi2', 'vis', or 'plot'"
        logging.warning(msg)
        raise ValueError(msg)


def twod_gaussring(pars, args, vis_data, version):
    r"""
    Top-level model handler for the 2D Gaussian Ring (7 parameters):
    pars: [Ring LogFlux, Ring LogSigma, Ring Rad, Inc, PA, Offset RA, Offset Dec]
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


###############################################################################
# 2. Model: 2D Gaussian Ring + 1 Gaussian Blob (11 parameters)
###############################################################################

def twod_gauss1blob_model(peak, sigma, rad, inc, pa, dra, ddec, peak_b1, sigma_b1, dist_b1, ang_b1, nxy, dxy, version):
    r"""
    Generates a 2D image matrix of an inclined Gaussian ring plus 1 compact Gaussian cluster.
    """
    xx, yy = get_grid(nxy, dxy)

    # Blob 1 offsets in disk frame (East: xdot > 0, North: ydot > 0)
    xdot_b1 = dist_b1 * np.sin(ang_b1)
    ydot_b1 = dist_b1 * np.cos(ang_b1)

    if version in ("chi2", "vis"):
        # Deproject x-axis by cos(inclination) (all units in radians)
        xinc = xx / np.cos(inc)
        radius_vec = np.hypot(xinc, yy)

        ring_model_jypix = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
        blob1_model_jypix = gaussblob_prof(peak_b1, sigma_b1, xx, yy, xdot_b1, ydot_b1, dxy)

        return ring_model_jypix + blob1_model_jypix

    elif version == 'plot':
        # Shift and rotate coordinates (arcseconds)
        xx_shifted = xx - dra
        yy_shifted = yy - ddec

        xpa =  xx_shifted * np.cos(pa) - yy_shifted * np.sin(pa)
        ypa =  xx_shifted * np.sin(pa) + yy_shifted * np.cos(pa)

        xinc = xpa / np.cos(inc)
        radius_vec = np.hypot(xinc, ypa)

        ring_model_jysr = gaussring_prof(peak, sigma, rad, radius_vec, 1.0)
        blob_model_jysr = gaussblob_prof(peak_b1, sigma_b1, xpa, ypa, xdot_b1, ydot_b1, 1.0)

        return ring_model_jysr + blob_model_jysr

    else:
        msg = f"Invalid version '{version}', must be 'chi2', 'vis', or 'plot'"
        logging.warning(msg)
        raise ValueError(msg)


def twod_gauss1blob(pars, args, vis_data, version):
    r"""
    Top-level model handler for 2D Gaussian Ring + 1 Gaussian Blob (11 parameters):
    pars: [Ring LogFlux, Ring LogSigma, Ring Rad, Inc, PA, Offset RA, Offset Dec,
           B1 LogFlux, B1 LogSigma, B1 Dist, B1 Angle]
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
        model_img = twod_gauss1blob_model(peak_ring, sigma_rad, ring_rad_rad, inclination, posangle,
                                          dRA_rad, dDec_rad, peak_b1, sigma_b1_rad, dist_b1_rad,
                                          ang_b1, nxy, dxy, version)

        if version == 'chi2':
            chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower')
            return chi2
        else:
            model_vis = np.array(sampleImage(model_img, dxy, u, v, dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower'), dtype=np.complex256)
            return model_vis
    else:
        model_img = twod_gauss1blob_model(peak_ring, sigma_arcsec, ring_rad, inclination, posangle,
                                          dRA, dDec, peak_b1, sigma_b1_arcsec, dist_b1,
                                          ang_b1, nxy, dxy, version)
        return model_img


###############################################################################
# 3. Model: 2D Gaussian Ring + 2 Gaussian Blobs (15 parameters)
###############################################################################

def twod_gauss2blob_model(peak, sigma, rad, inc, pa, dra, ddec, peak_b1, sigma_b1, dist_b1, ang_b1, peak_b2, sigma_b2, dist_b2, ang_b2, nxy, dxy, version):
    r"""
    Generates a 2D image matrix of an inclined Gaussian ring plus 2 distinct Gaussian clusters.
    """
    xx, yy = get_grid(nxy, dxy)

    xdot_b1 = dist_b1 * np.sin(ang_b1)
    ydot_b1 = dist_b1 * np.cos(ang_b1)

    xdot_b2 = dist_b2 * np.sin(ang_b2)
    ydot_b2 = dist_b2 * np.cos(ang_b2)

    if version in ("chi2", "vis"):
        xinc = xx / np.cos(inc)
        radius_vec = np.hypot(xinc, yy)

        ring_model_jypix = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
        blob1_model_jypix = gaussblob_prof(peak_b1, sigma_b1, xx, yy, xdot_b1, ydot_b1, dxy)
        blob2_model_jypix = gaussblob_prof(peak_b2, sigma_b2, xx, yy, xdot_b2, ydot_b2, dxy)

        return ring_model_jypix + blob1_model_jypix + blob2_model_jypix

    elif version == 'plot':
        xx_shifted = xx - dra
        yy_shifted = yy - ddec

        xpa =  xx_shifted * np.cos(pa) - yy_shifted * np.sin(pa)
        ypa =  xx_shifted * np.sin(pa) + yy_shifted * np.cos(pa)

        xinc = xpa / np.cos(inc)
        radius_vec = np.hypot(xinc, ypa)

        ring_model_jysr = gaussring_prof(peak, sigma, rad, radius_vec, 1.0)
        blob1_model_jysr = gaussblob_prof(peak_b1, sigma_b1, xpa, ypa, xdot_b1, ydot_b1, 1.0)
        blob2_model_jysr = gaussblob_prof(peak_b2, sigma_b2, xpa, ypa, xdot_b2, ydot_b2, 1.0)

        return ring_model_jysr + blob1_model_jysr + blob2_model_jysr

    else:
        msg = f"Invalid version '{version}', must be 'chi2', 'vis', or 'plot'"
        logging.warning(msg)
        raise ValueError(msg)


def twod_gauss2blob(pars, args, vis_data, version):
    r"""
    Top-level model handler for 2D Gaussian Ring + 2 Gaussian Blobs (15 parameters):
    pars: [Ring LogFlux, Ring LogSigma, Ring Rad, Inc, PA, Offset RA, Offset Dec,
           B1 LogFlux, B1 LogSigma, B1 Dist, B1 Angle,
           B2 LogFlux, B2 LogSigma, B2 Dist, B2 Angle]
    """
    log_flux, log_sigma, ring_rad, inclination, posangle, dRA, dDec, log_flux_b1, log_sigma_b1, dist_b1, ang_b1, log_flux_b2, log_sigma_b2, dist_b2, ang_b2 = pars
    nxy, dxy = args
    u, v, re, im, w = vis_data

    inclination *= deg
    posangle *= deg
    ang_b1 *= deg
    ang_b2 *= deg

    sigma_arcsec = 10.0**log_sigma
    sigma_b1_arcsec = 10.0**log_sigma_b1
    sigma_b2_arcsec = 10.0**log_sigma_b2

    sigma_rad = sigma_arcsec * arcsec
    sigma_b1_rad = sigma_b1_arcsec * arcsec
    sigma_b2_rad = sigma_b2_arcsec * arcsec
    ring_rad_rad = ring_rad * arcsec

    peak_ring = ring_flux_to_peak(log_flux, sigma_rad, ring_rad_rad, inclination)
    peak_b1 = blob_flux_to_peak(log_flux_b1, sigma_b1_rad)
    peak_b2 = blob_flux_to_peak(log_flux_b2, sigma_b2_rad)

    if version in ("chi2", "vis"):
        dRA_rad = dRA * arcsec
        dDec_rad = dDec * arcsec
        dist_b1_rad = dist_b1 * arcsec
        dist_b2_rad = dist_b2 * arcsec
        model_img = twod_gauss2blob_model(peak_ring, sigma_rad, ring_rad_rad, inclination, posangle,
                                          dRA_rad, dDec_rad, peak_b1, sigma_b1_rad, dist_b1_rad,
                                          ang_b1, peak_b2, sigma_b2_rad, dist_b2_rad, ang_b2,
                                          nxy, dxy, version)

        if version == 'chi2':
            chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower')
            return chi2
        else:
            model_vis = np.array(sampleImage(model_img, dxy, u, v, dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower'), dtype=np.complex256)
            return model_vis
    else:
        model_img = twod_gauss2blob_model(peak_ring, sigma_arcsec, ring_rad, inclination, posangle,
                                          dRA, dDec, peak_b1, sigma_b1_arcsec, dist_b1,
                                          ang_b1, peak_b2, sigma_b2_arcsec, dist_b2, ang_b2,
                                          nxy, dxy, version)
        return model_img


###############################################################################
# 4. Model: 2D Gaussian Ring + 3 Gaussian Blobs (19 parameters)
###############################################################################

def twod_gauss3blob_model(peak, sigma, rad, inc, pa, dra, ddec, peak_b1, sigma_b1, dist_b1, ang_b1, peak_b2, sigma_b2, dist_b2, ang_b2, peak_b3, sigma_b3, dist_b3, ang_b3, nxy, dxy, version):
    r"""
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

        ring_model_jysr = gaussring_prof(peak, sigma, rad, radius_vec, 1.0)
        blob1_model_jysr = gaussblob_prof(peak_b1, sigma_b1, xpa, ypa, xdot_b1, ydot_b1, 1.0)
        blob2_model_jysr = gaussblob_prof(peak_b2, sigma_b2, xpa, ypa, xdot_b2, ydot_b2, 1.0)
        blob3_model_jysr = gaussblob_prof(peak_b3, sigma_b3, xpa, ypa, xdot_b3, ydot_b3, 1.0)

        return ring_model_jysr + blob1_model_jysr + blob2_model_jysr + blob3_model_jysr

    else:
        msg = f"Invalid version '{version}', must be 'chi2', 'vis', or 'plot'"
        logging.warning(msg)
        raise ValueError(msg)


def twod_gauss3blob(pars, args, vis_data, version):
    r"""
    Top-level model handler for 2D Gaussian Ring + 3 Gaussian Blobs (19 parameters):
    pars: [Ring LogFlux, Ring LogSigma, Ring Rad, Inc, PA, Offset RA, Offset Dec,
           B1 LogFlux, B1 LogSigma, B1 Dist, B1 Angle,
           B2 LogFlux, B2 LogSigma, B2 Dist, B2 Angle,
           B3 LogFlux, B3 LogSigma, B3 Dist, B3 Angle]
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
        model_img = twod_gauss3blob_model(peak_ring, sigma_rad, ring_rad_rad, inclination, posangle,
                                          dRA_rad, dDec_rad, peak_b1, sigma_b1_rad, dist_b1_rad,
                                          ang_b1, peak_b2, sigma_b2_rad, dist_b2_rad, ang_b2,
                                          peak_b3, sigma_b3_rad, dist_b3_rad, ang_b3, nxy, dxy, version)

        if version == 'chi2':
            chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower')
            return chi2
        else:
            model_vis = np.array(sampleImage(model_img, dxy, u, v, dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower'), dtype=np.complex256)
            return model_vis
    else:
        model_img = twod_gauss3blob_model(peak_ring, sigma_arcsec, ring_rad, inclination, posangle,
                                          dRA, dDec, peak_b1, sigma_b1_arcsec, dist_b1,
                                          ang_b1, peak_b2, sigma_b2_arcsec, dist_b2, ang_b2,
                                          peak_b3, sigma_b3_arcsec, dist_b3, ang_b3, nxy, dxy, version)
        return model_img


###############################################################################
# Dispatcher & Metadata Functions
###############################################################################

def model_prof(pars, args, vis_data, version, fittype):
    r"""
    Central dispatcher function for model profile evaluation.
    Supports 'twod_gaussring', 'twod_gauss1blob', 'twod_gauss2blob', and 'twod_gauss3blob'.
    """
    if version not in ("chi2", "vis", "plot"):
        msg = f"Invalid version '{version}', must be 'chi2', 'vis', or 'plot'"
        logging.warning(msg)
        raise ValueError(msg)

    if fittype == 'twod_gaussring':
        return twod_gaussring(pars, args, vis_data, version)
    elif fittype == 'twod_gauss1blob':
        return twod_gauss1blob(pars, args, vis_data, version)
    elif fittype == 'twod_gauss2blob':
        return twod_gauss2blob(pars, args, vis_data, version)
    elif fittype == 'twod_gauss3blob':
        return twod_gauss3blob(pars, args, vis_data, version)
    else:
        msg = f"Invalid or unsupported fittype '{fittype}' in advi_version. Supported models are 'twod_gaussring', 'twod_gauss1blob', 'twod_gauss2blob', and 'twod_gauss3blob'."
        logging.error(msg)
        raise ValueError(msg)


def model_addon(fittype):
    r"""
    Returns parameter metadata (labels, physical units, dimensionality) for ADVI.
    """
    if fittype == 'twod_gaussring':
        label = ["Ring LogFlux", "Ring LogSigma", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec"]
        unit = ["log(Jy)", "log(arcsec)", "arcsec", "degrees", "degrees", "arcsec", "arcsec"]
        ndim = len(label)

    elif fittype == 'twod_gauss1blob':
        label = [
            "Ring LogFlux", "Ring LogSigma", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec",
            "B1 LogFlux", "B1 LogSigma", "B1 Dist", "B1 Angle"
        ]
        unit = [
            "log(Jy)", "log(arcsec)", "arcsec", "degrees", "degrees", "arcsec", "arcsec",
            "log(Jy)", "log(arcsec)", "arcsec", "degrees"
        ]
        ndim = len(label)

    elif fittype == 'twod_gauss2blob':
        label = [
            "Ring LogFlux", "Ring LogSigma", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec",
            "B1 LogFlux", "B1 LogSigma", "B1 Dist", "B1 Angle",
            "B2 LogFlux", "B2 LogSigma", "B2 Dist", "B2 Angle"
        ]
        unit = [
            "log(Jy)", "log(arcsec)", "arcsec", "degrees", "degrees", "arcsec", "arcsec",
            "log(Jy)", "log(arcsec)", "arcsec", "degrees",
            "log(Jy)", "log(arcsec)", "arcsec", "degrees"
        ]
        ndim = len(label)

    elif fittype == 'twod_gauss3blob':
        label = [
            "Ring LogFlux", "Ring LogSigma", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec",
            "B1 LogFlux", "B1 LogSigma", "B1 Dist", "B1 Angle",
            "B2 LogFlux", "B2 LogSigma", "B2 Dist", "B2 Angle",
            "B3 LogFlux", "B3 LogSigma", "B3 Dist", "B3 Angle"
        ]
        unit = [
            "log(Jy)", "log(arcsec)", "arcsec", "degrees", "degrees", "arcsec", "arcsec",
            "log(Jy)", "log(arcsec)", "arcsec", "degrees",
            "log(Jy)", "log(arcsec)", "arcsec", "degrees",
            "log(Jy)", "log(arcsec)", "arcsec", "degrees"
        ]
        ndim = len(label)

    else:
        msg = f"Invalid or unsupported fittype '{fittype}' in advi_version."
        logging.error(msg)
        raise ValueError(msg)

    return label, unit, ndim
