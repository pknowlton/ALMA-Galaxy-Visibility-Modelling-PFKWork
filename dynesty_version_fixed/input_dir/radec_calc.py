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

    Parameters
    ----------
    pars : array_like
        Best-fit model parameters.
    fittype : str
        Model identifier: 'twod_gaussring', 'twod_gauss1blob', 'twod_gauss1blob_2peak',
        'twod_gauss1blob_2peak_dp', 'twod_gauss2blob', or 'twod_gauss3blob'.

    Returns
    -------
    dynest_coords : numpy.ndarray, shape (N, 2)
        Array of (RA, Dec) coordinates in degrees for each modeled clump / component.
    """
    logging.info('Fittype: %s', fittype)

    # ALMA observation phase center
    phase_cent = SkyCoord('10h43m57.7330s', '+11d42m12.9996s', frame='icrs')
    
    if fittype == 'twod_gaussring':
        # Fitted ring centroid offset
        log_flux, log_sigma, ring_rad, inclination, posangle, dRA, dDec = pars
        ngc3351 = phase_cent.spherical_offsets_by(dRA * u.arcsec, dDec * u.arcsec)
        dynest_coords = np.zeros((1, 2))
        dynest_coords[0, 0] = ngc3351.ra.deg
        dynest_coords[0, 1] = ngc3351.dec.deg

    elif fittype == 'twod_gauss1blob':
        log_flux, log_sigma, ring_rad, inclination, posangle, dRA, dDec, log_flux_b1, log_sigma_b1, dist_b1, ang_b1 = pars
        ngc3351 = phase_cent.spherical_offsets_by(dRA * u.arcsec, dDec * u.arcsec)
        dynest_coords = np.zeros((1, 2))

        # Position angle on the sky: (posangle + ang_b1) East of North
        sky_ang_b1 = (posangle + ang_b1) * u.deg
        sep_b1 = dist_b1 * u.arcsecond

        radec_b1 = ngc3351.directional_offset_by(position_angle=sky_ang_b1, separation=sep_b1)
        dynest_coords[0, 0] = radec_b1.ra.deg
        dynest_coords[0, 1] = radec_b1.dec.deg

    elif fittype == 'twod_gauss1blob_2peak':
        log_flux, log_sigma, ring_rad, inclination, posangle, dRA, dDec, log_flux_b11, log_sigma_b11, log_flux_b12, log_sigma_b12, dist_b1, ang_b1 = pars
        ngc3351 = phase_cent.spherical_offsets_by(dRA * u.arcsec, dDec * u.arcsec)
        dynest_coords = np.zeros((1, 2))

        sky_ang_b1 = (posangle + ang_b1) * u.deg
        sep_b1 = dist_b1 * u.arcsecond

        radec_b1 = ngc3351.directional_offset_by(position_angle=sky_ang_b1, separation=sep_b1)
        dynest_coords[0, 0] = radec_b1.ra.deg
        dynest_coords[0, 1] = radec_b1.dec.deg

    elif fittype == 'twod_gauss1blob_2peak_dp':
        log_flux, log_sigma, ring_rad, inclination, posangle, dRA, dDec, log_flux_b11, log_sigma_b11, dist_b11, ang_b11, log_flux_b12, log_sigma_b12, dist_b12, ang_b12 = pars
        ngc3351 = phase_cent.spherical_offsets_by(dRA * u.arcsec, dDec * u.arcsec)
        dynest_coords = np.zeros((2, 2))

        # Peak 1 relative to galaxy center
        sky_ang_b11 = (posangle + ang_b11) * u.deg
        sep_b11 = dist_b11 * u.arcsecond
        radec_b11 = ngc3351.directional_offset_by(position_angle=sky_ang_b11, separation=sep_b11)
        dynest_coords[0, 0] = radec_b11.ra.deg
        dynest_coords[0, 1] = radec_b11.dec.deg

        # Peak 2 relative to Peak 1
        sky_ang_b12 = (posangle + ang_b12) * u.deg
        sep_b12 = dist_b12 * u.arcsecond
        radec_b12 = radec_b11.directional_offset_by(position_angle=sky_ang_b12, separation=sep_b12)
        dynest_coords[1, 0] = radec_b12.ra.deg
        dynest_coords[1, 1] = radec_b12.dec.deg

    elif fittype == 'twod_gauss2blob':
        log_flux, log_sigma, ring_rad, inclination, posangle, dRA, dDec, log_flux_b1, log_sigma_b1, dist_b1, ang_b1, log_flux_b2, log_sigma_b2, dist_b2, ang_b2 = pars
        ngc3351 = phase_cent.spherical_offsets_by(dRA * u.arcsec, dDec * u.arcsec)
        dynest_coords = np.zeros((2, 2))

        for idx, (dist_b, ang_b) in enumerate([(dist_b1, ang_b1), (dist_b2, ang_b2)]):
            sky_ang = (posangle + ang_b) * u.deg
            sep = dist_b * u.arcsecond
            radec = ngc3351.directional_offset_by(position_angle=sky_ang, separation=sep)
            dynest_coords[idx, 0] = radec.ra.deg
            dynest_coords[idx, 1] = radec.dec.deg

    elif fittype == 'twod_gauss3blob':
        log_flux, log_sigma, ring_rad, inclination, posangle, dRA, dDec, log_flux_b1, log_sigma_b1, dist_b1, ang_b1, log_flux_b2, log_sigma_b2, dist_b2, ang_b2, log_flux_b3, log_sigma_b3, dist_b3, ang_b3 = pars
        ngc3351 = phase_cent.spherical_offsets_by(dRA * u.arcsec, dDec * u.arcsec)
        dynest_coords = np.zeros((3, 2))

        for idx, (dist_b, ang_b) in enumerate([(dist_b1, ang_b1), (dist_b2, ang_b2), (dist_b3, ang_b3)]):
            sky_ang = (posangle + ang_b) * u.deg
            sep = dist_b * u.arcsecond
            radec = ngc3351.directional_offset_by(position_angle=sky_ang, separation=sep)
            dynest_coords[idx, 0] = radec.ra.deg
            dynest_coords[idx, 1] = radec.dec.deg

    elif fittype in ('simgauss', 'twod_simgauss'):
        b_p1, b_p2, dist, ang, posangle, dRA, dDec = pars
        ngc3351 = phase_cent.spherical_offsets_by(dRA * u.arcsec, dDec * u.arcsec)
        dynest_coords = np.zeros((1, 2))

        sky_ang = (posangle + ang) * u.deg
        sep = dist * u.arcsecond
        radec = ngc3351.directional_offset_by(position_angle=sky_ang, separation=sep)
        dynest_coords[0, 0] = radec.ra.deg
        dynest_coords[0, 1] = radec.dec.deg

    else:
        msg = f"Invalid fittype '{fittype}', please choose a valid fitting model."
        logging.warning(msg)
        raise ValueError(msg)

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
