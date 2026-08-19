import numpy as np
import logging
from galario.double import get_image_size, deg, arcsec, chi2Profile, chi2Image, sampleProfile, sampleImage
import pandas as pd

#########################
### Base profile functions
#########################

def gaussring_prof(peak_ring, sigma_ring, rad_ring, radius, dxy):
    return 10**peak_ring * np.exp((-1/2)*((radius-rad_ring)/sigma_ring)**2) * (dxy**2)

def gaussblob_prof(peak, sigma, xx, yy, xoff, yoff, dxy):
    return 10**peak * np.exp((-1/2) * (((xx-xoff)/sigma)**2 + ((yy-yoff)/sigma)**2)) * (dxy**2)

PA_RAD = 15.0 * deg
COS_PA = np.cos(PA_RAD)
SIN_PA = np.sin(PA_RAD)

#########################
### 2D Gaussian Ring
#########################

def twod_gaussring_model(peak, sigma, rad, inc, pa, dra, ddec, nxy, dxy, version):

    #Initialize the image plane
    image_size = nxy * dxy
    x = np.linspace(-image_size/2, image_size/2, nxy)
    y = np.linspace(-image_size/2, image_size/2, nxy)
    xx, yy = np.meshgrid(x, y)

    if version in ("chi2", "vis"):

        xinc = xx/np.cos(inc)
        radius_vec = np.hypot(xinc, yy)

        ring_model_jypix = gaussring_prof(peak, sigma, rad, radius_vec, dxy)

        return ring_model_jypix

    elif version=='plot':

        xx_shifted = xx + (dra)
        yy_shifted = yy - (ddec)

        xpa = xx_shifted*np.cos(pa) + yy_shifted*np.sin(pa)
        ypa = -xx_shifted*np.sin(pa) + yy_shifted*np.cos(pa)

        xinc = xpa/np.cos(inc)
        radius_vec = np.hypot(xinc, ypa)

        ring_model_jysr = gaussring_prof(peak, sigma, rad, radius_vec, 1)

        return ring_model_jysr

    else:

        msg = f"Invalid version '{version}', must be 'chi2', 'vis', or 'plot'"
        logging.warning(msg)
        raise ValueError(msg)

def twod_gaussring(pars, args, vis_data, version):

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec = pars
    nxy, dxy = args
    u, v, re, im, w = vis_data

    inclination *= deg
    posangle *= deg

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

    #Initialize the image plane
    image_size = nxy * dxy
    x = np.linspace(-image_size/2, image_size/2, nxy)
    y = np.linspace(-image_size/2, image_size/2, nxy)
    xx, yy = np.meshgrid(x, y)

    if version in ("chi2", "vis"):

        #best guess position angle applied
        xpa_rg = xx*COS_PA + yy*SIN_PA
        ypa_rg = -xx*SIN_PA + yy*COS_PA
        
        xinc = xpa_rg/np.cos(inc)
        radius_vec = np.hypot(xinc, ypa_rg)

        xdot_b1 = dist_b1 * np.sin(ang_b1)
        ydot_b1 = dist_b1 * np.cos(ang_b1)

        ring_model_jypix = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
        blob1_model_jypix = gaussblob_prof(peak_b1, sigma_b1, xx, yy, xdot_b1, ydot_b1, dxy)

        return ring_model_jypix +  blob1_model_jypix

    elif version=='plot':

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

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b1, sigma_b1, dist_b1, ang_b1 = pars
    nxy, dxy = args
    u, v, re, im, w = vis_data

    inclination *= deg
    posangle *= deg
    ang_b1 *= deg

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


##################################################################################################################################################################################


def model_prof(pars, args, vis_data, version, fittype):

    if version not in ("chi2", "vis", "plot"):

        msg = f"Invalid version '{version}', must be 'chi2', 'vis', or 'plot'"
        logging.warning(msg)
        raise ValueError(msg)


    if fittype == 'twod_gaussring':
        prof = twod_gaussring(pars, args, vis_data, version)

    elif fittype == 'twod_gauss1blob':
        prof = twod_gauss1blob(pars, args, vis_data, version)

    else:
        msg = f"Invalid fittype '{fittype}', please choose a valid fitting model, or add a new one into the code."
        logging.warning(msg)
        raise ValueError(msg)

    return prof

##################################################################################################################################################################################
##################################################################################################################################################################################
##################################################################################################################################################################################


def model_addon(fittype):

    if fittype == 'twod_gaussring':
        label = ["Peak", "Width", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec"]
        unit = ["log(Jy/sr)", "arcsec", "arcsec", "degrees", "degrees", "arcsec", "arcsec"]
        ndim = len(label)

    elif fittype == 'twod_gauss1blob':
        label = ["Peak", "Width", "Offset", "Inc", "PA", "Offset RA", "Offset Dec", "B. Peak", "B. Width", "Dist", "Angle"]
        unit = ["log(Jy/sr)", "arcsec", "arcsec", "degrees", "degrees", "arcsec", "arcsec", "log(Jy/sr)", "arcsec", "arcsec", "degrees"]
        ndim = len(label)


    else:
        msg = f"Invalid fittype '{fittype}', please choose a valid fitting model, or add a new one into the code."
        logging.warning(msg)
        raise ValueError(msg)

    return label, unit, ndim