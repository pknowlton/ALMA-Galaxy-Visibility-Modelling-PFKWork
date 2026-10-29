"""
Celestial Coordinate Conversions and Model WCS Helpers
======================================================
This module handles transformations between internal model parameters (offsets,
distances, position angles) and celestial sky coordinates (Right Ascension and Declination
in ICRS J2000), as well as generating Astropy World Coordinate System (WCS) instances.

Conventions:
------------
- Right Ascension offsets (dRA) are positive towards the East.
- Declination offsets (dDec) are positive towards the North.
- Disk Position Angle (posangle) is defined East of North.
- Blob azimuthal angles (ang) are defined East of North in the disk frame, so the
  resultant position angle on the sky is (posangle + ang).
"""

import numpy as np
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd
import logging
from model_defs import get_spec
from config import PHASE_CENTER_RA, PHASE_CENTER_DEC

def sun_radec(ymc_ids):
    """
    Extracts reference coordinates for Young Massive Clusters (YMCs) from Sun et al. (2024).

    Parameters
    ----------
    ymc_ids : list of int
        Cluster identifier numbers (e.g. [15, 6, 18]).

    Returns
    -------
    sun_coords : numpy.ndarray, shape (N, 2)
        Array of (RA, Dec) coordinates in degrees.
    """
    logging.info(f"YMC IDs: {ymc_ids}")
    sun_coords = np.zeros((len(ymc_ids), 2))
    ymcs = pd.read_csv('ymc_prior_full_err.csv').set_index('ymc_id')

    for i, ymc_id in enumerate(ymc_ids):
        sun_coords[i, 0] = ymcs.loc[ymc_id, 'ra (deg)']
        sun_coords[i, 1] = ymcs.loc[ymc_id, 'dec (deg)']

    logging.info(f"YMC Coords (Sun et al. 2024): {sun_coords}")
    return sun_coords

def dynest_radec(pars, fittype):
    """
    Computes celestial (RA, Dec) coordinates for modeled clumps / components.

    For ring-only models the single returned position is the fitted ring centroid.
    For models with blobs, one position is returned per blob, in blob order.
    The disk geometry (position angle, centroid offsets) comes from the ring
    parameters, from the fixed best-fit constants (residual models), or from the
    free geometry parameters (simgauss); see `model_defs.py`.

    Parameters
    ----------
    pars : array_like
        Best-fit model parameters.
    fittype : str
        Model identifier registered in `model_defs.py` (e.g. 'twod_gauss3blob').

    Returns
    -------
    dynest_coords : numpy.ndarray, shape (N, 2)
        Array of (RA, Dec) coordinates in degrees for each modeled clump / component.
    """
    logging.info('Fittype: %s', fittype)
    spec = get_spec(fittype)

    # ALMA observation phase center (set in config.py)
    phase_cent = SkyCoord(PHASE_CENTER_RA, PHASE_CENTER_DEC, frame='icrs')

    posangle, dRA, dDec = spec.geometry_params(pars)
    ngc3351 = phase_cent.spherical_offsets_by(dRA * u.arcsec, dDec * u.arcsec)

    if spec.n_blobs == 0:
        # Ring-only model: report the fitted ring centroid
        dynest_coords = np.zeros((1, 2))
        dynest_coords[0, 0] = ngc3351.ra.deg
        dynest_coords[0, 1] = ngc3351.dec.deg
    else:
        dynest_coords = np.zeros((spec.n_blobs, 2))
        for k in range(spec.n_blobs):
            i = spec.blob_index(k)
            dist_b, ang_b = pars[i + 2], pars[i + 3]
            # Position angle on the sky: (posangle + ang) East of North
            sky_ang = (posangle + ang_b) * u.deg
            sep = dist_b * u.arcsecond
            radec = ngc3351.directional_offset_by(position_angle=sky_ang, separation=sep)
            dynest_coords[k, 0] = radec.ra.deg
            dynest_coords[k, 1] = radec.dec.deg

    logging.info(f"Dynesty Coords: {dynest_coords}")
    return dynest_coords

def make_model_wcs(ra_center, dec_center, pixel_scale_arcsec, shape, projection="SIN"):
    """
    Constructs an Astropy WCS instance matching Galario image coordinate grids.

    Parameters
    ----------
    ra_center : float
        Right Ascension of image reference pixel in degrees.
    dec_center : float
        Declination of image reference pixel in degrees.
    pixel_scale_arcsec : float
        Angular pixel scale in arcseconds per pixel.
    shape : tuple of int
        Image shape (ny, nx).
    projection : str
        FITS projection type (default: 'SIN' for interferometric synthesis).

    Returns
    -------
    astropy.wcs.WCS
        WCS object configured for the model image.
    """
    ny, nx = shape
    pixel_scale_deg = pixel_scale_arcsec / 3600.0
    
    wcs = WCS(naxis=2)
    wcs.array_shape = (ny, nx)
    wcs.wcs.crpix = [(nx / 2.0) + 1, (ny / 2.0) + 1]
    wcs.wcs.crval = [ra_center, dec_center]
    wcs.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]  # RA increases to the left (negative CDELT)
    wcs.wcs.ctype = [f"RA---{projection}", f"DEC--{projection}"]
    wcs.wcs.cunit = ["deg", "deg"]
    
    return wcs
