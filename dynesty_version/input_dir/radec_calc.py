import numpy as np
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd
import logging

def sun_radec(ymc_ids):

    logging.info(f"YMC IDs: {ymc_ids}")

    sun_coords = np.zeros((len(ymc_ids), 2))
    ymcs = pd.read_csv('ymc_prior_full_err.csv').set_index('ymc_id')

    for i, ymc_id in enumerate(ymc_ids):

        sun_coords[i, 0] = ymcs.loc[ymc_id, 'ra (deg)']
        sun_coords[i, 1] = ymcs.loc[ymc_id, 'dec (deg)']

    logging.info(f"YMC Coords (Sun et al. 2024): {sun_coords}")

    return sun_coords

def dynest_radec(pars, fittype):

    logging.info('Fittype: %s', fittype)

    ngc3351 = SkyCoord('10h43m57.75s', '+11d42m13.34s', frame='icrs')

    if fittype == 'twod_gauss1blob':
        
        peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b1, sigma_b1, dist_b1, ang_b1 = pars

        #ngc3351 = ngc3351_old.spherical_offsets_by(dRA * u.arcsec, dDec * u.arcsec)

        dynest_coords = np.zeros((1, 2))

        co_ang_b1 = ang_b1 - posangle
        ap_ang_b1 = (360 - co_ang_b1) * u.deg
        ap_dist_b1 = dist_b1 * u.arcsecond

        radec_b1 = ngc3351.directional_offset_by(position_angle=ap_ang_b1, separation=ap_dist_b1)
        dynest_coords[0, 0] = radec_b1.ra.deg
        dynest_coords[0, 1] = radec_b1.dec.deg

    elif fittype == 'twod_gauss1blob_2peak':
        
        peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b11, sigma_b11, peak_b12, sigma_b12, dist_b1, ang_b1 = pars

        dynest_coords = np.zeros((1, 2))

        co_ang_b1 = ang_b1 - posangle
        ap_ang_b1 = (360 - co_ang_b1) * u.deg
        ap_dist_b1 = dist_b1 * u.arcsecond

        radec_b1 = ngc3351.directional_offset_by(position_angle=ap_ang_b1, separation=ap_dist_b1)
        dynest_coords[0, 0] = radec_b1.ra.deg
        dynest_coords[0, 1] = radec_b1.dec.deg

    elif fittype == 'twod_gauss3blob':
        
        peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b1, sigma_b1, dist_b1, ang_b1, peak_b2, sigma_b2, dist_b2, ang_b2, peak_b3, sigma_b3, dist_b3, ang_b3 = pars

        dynest_coords = np.zeros((3, 2))

        co_ang_b1 = ang_b1 - posangle
        ap_ang_b1 = (360 - co_ang_b1) * u.deg
        ap_dist_b1 = dist_b1 * u.arcsecond

        radec_b1 = ngc3351.directional_offset_by(position_angle=ap_ang_b1, separation=ap_dist_b1)
        dynest_coords[0, 0] = radec_b1.ra.deg
        dynest_coords[0, 1] = radec_b1.dec.deg

        co_ang_b2 = ang_b2 - posangle
        ap_ang_b2 = (360 - co_ang_b2) * u.deg
        ap_dist_b2 = dist_b2 * u.arcsecond

        radec_b2 = ngc3351.directional_offset_by(position_angle=ap_ang_b2, separation=ap_dist_b2)
        dynest_coords[1, 0] = radec_b2.ra.deg
        dynest_coords[1, 1] = radec_b2.dec.deg

        co_ang_b3 = ang_b3 - posangle
        ap_ang_b3 = (360 - co_ang_b3) * u.deg
        ap_dist_b3 = dist_b3 * u.arcsecond

        radec_b3 = ngc3351.directional_offset_by(position_angle=ap_ang_b3, separation=ap_dist_b3)
        dynest_coords[2, 0] = radec_b3.ra.deg
        dynest_coords[2, 1] = radec_b3.dec.deg

    else:
        msg = f"Invalid fittype '{fittype}', please choose a valid fitting model, or add a new one into the code."
        logging.warning(msg)
        raise ValueError(msg)

    logging.info(f"Dynesty Coords: {dynest_coords}")

    return dynest_coords

def make_model_wcs(ra_center, dec_center, pixel_scale_arcsec, shape, projection="TAN"):

    ny, nx = shape
    pixel_scale_deg = pixel_scale_arcsec / 3600.0
    
    wcs = WCS(naxis=2)
    # Reference pixel (1-indexed FITS convention; center of pixel array)
    wcs.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    # Reference world coordinate at CRPIX
    wcs.wcs.crval = [ra_center, dec_center]
    # Pixel scale: RA step is negative so RA increases to the left
    wcs.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]
    # Coordinate types and units
    wcs.wcs.ctype = [f"RA---{projection}", f"DEC--{projection}"]
    wcs.wcs.cunit = ["deg", "deg"]
    
    return wcs