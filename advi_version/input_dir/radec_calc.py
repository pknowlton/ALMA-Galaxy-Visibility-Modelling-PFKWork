r"""
Celestial Coordinate Conversions and Model WCS Helpers for ADVI
================================================================
This module handles transformations between internal model parameters (offsets,
distances, position angles) and celestial sky coordinates (Right Ascension and Declination
in ICRS J2000), as well as generating Astropy World Coordinate System (WCS) instances.

Conventions:
- Right Ascension offsets (dRA) are positive towards East.
- Declination offsets (dDec) are positive towards North.
- Disk Position Angle (posangle) is defined East of North.
- Blob azimuthal angles (ang) are defined East of North in the disk frame.
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
    """
    logging.info(f"YMC IDs: {ymc_ids}")
    sun_coords = np.zeros((len(ymc_ids), 2))
    ymcs = pd.read_csv('ymc_prior_full_err.csv').set_index('ymc_id')

    for i, ymc_id in enumerate(ymc_ids):
        sun_coords[i, 0] = ymcs.loc[ymc_id, 'ra (deg)']
        sun_coords[i, 1] = ymcs.loc[ymc_id, 'dec (deg)']

    logging.info(f"YMC Coords (Sun et al. 2024): {sun_coords}")
    return sun_coords


def advi_radec(pars, fittype):
    """
    Computes celestial (RA, Dec) coordinates for modeled clumps / components.
    Supports 'twod_gaussring' and 'twod_gauss1blob'.
    """
    logging.info('Fittype: %s', fittype)

    # ALMA observation phase center
    phase_cent = SkyCoord('10h43m57.7330s', '+11d42m12.9996s', frame='icrs')

    if fittype == 'twod_gaussring':
        log_flux, log_sigma, ring_rad, inclination, posangle, dRA, dDec = pars
        ngc3351 = phase_cent.spherical_offsets_by(dRA * u.arcsec, dDec * u.arcsec)
        coords = np.zeros((1, 2))
        coords[0, 0] = ngc3351.ra.deg
        coords[0, 1] = ngc3351.dec.deg

    elif fittype == 'twod_gauss1blob':
        log_flux, log_sigma, ring_rad, inclination, posangle, dRA, dDec, log_flux_b1, log_sigma_b1, dist_b1, ang_b1 = pars
        ngc3351 = phase_cent.spherical_offsets_by(dRA * u.arcsec, dDec * u.arcsec)
        coords = np.zeros((1, 2))

        # Position angle on sky: (posangle + ang_b1) East of North
        sky_ang_b1 = (posangle + ang_b1) * u.deg
        sep_b1 = dist_b1 * u.arcsecond

        radec_b1 = ngc3351.directional_offset_by(position_angle=sky_ang_b1, separation=sep_b1)
        coords[0, 0] = radec_b1.ra.deg
        coords[0, 1] = radec_b1.dec.deg

    else:
        msg = f"Invalid fittype '{fittype}', please choose 'twod_gaussring' or 'twod_gauss1blob'."
        logging.warning(msg)
        raise ValueError(msg)

    logging.info(f"ADVI Model Coords: {coords}")
    return coords

# Backward compatibility alias
dynest_radec = advi_radec


def make_model_wcs(ra_center, dec_center, pixel_scale_arcsec, shape, projection="SIN"):
    """
    Constructs an Astropy WCS instance matching Galario image coordinate grids.
    """
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2.0 + 1.0, shape[0] / 2.0 + 1.0]
    cdelt_deg = pixel_scale_arcsec / 3600.0
    w.wcs.cdelt = [-cdelt_deg, cdelt_deg]
    w.wcs.crval = [ra_center, dec_center]
    w.wcs.ctype = [f"RA---{projection}", f"DEC--{projection}"]
    w.wcs.cunit = ["deg", "deg"]
    return w
