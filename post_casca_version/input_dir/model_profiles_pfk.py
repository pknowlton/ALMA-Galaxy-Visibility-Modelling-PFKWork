import numpy as np
import logging
from galario.double import get_image_size, chi2Profile, deg, arcsec, chi2Image

#########################
### Radial Gaussian Ring
#########################

def radial_gaussian_ring(pars, args, vis_data):

    """
    Calculates chi-squared for a Gaussian ring intensity profile.

    Converts input parameters from log-space and angular units (arcsec/deg) 
    to linear space and radians for the visibility-plane calculation.

    Args:
        pars (np.ndarray):1D array of model parameters to be sampled.
            [peak (log(Jy/sr)), sigma (arcsec), ring_rad (arcsec), 
             inc (deg), PA (deg), dRA (arcsec), dDec (arcsec)]
        args (tuple): Fixed data and constants required for the model.
            (start, step, numsteps, nxy, dxy, u, v, re, im, w)

    Returns:
        float: The chi-squared value calculated via chi2Profile from GALARIO.
    """

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec = pars
    start, step, numsteps, nxy, dxy = args
    u, v, re, im, w = vis_data

    # convert from log to real space
    peak = 10**peak   

    # convert to radians
    sigma *= arcsec
    ring_rad *= arcsec
    start *= arcsec
    step *= arcsec

    inclination *= deg
    posangle *= deg

    dRA *= arcsec
    dDec *= arcsec

    #define gaussian profile
    radius = np.linspace(start, start + numsteps*step, numsteps)

    rad_prof = peak*np.exp((-1/2)*((radius-ring_rad)/sigma)**2)

    #compute the chi-square of the model
    chi2 = chi2Profile(rad_prof, start, step, nxy, dxy, u, v, re, im, w, inc=inclination, PA=posangle, dRA=dRA, dDec=dDec)

    return chi2

#########################
### 2D Gaussian Ring
#########################

def gaussring_prof(peak_ring, sigma_ring, rad_ring, radius, dxy):
    return 10**peak_ring * np.exp((-1/2)*((radius-rad_ring)/sigma_ring)**2) * (dxy**2)

def gaussring_model(peak, sigma, rad, inc, nxy, dxy):
    #Initialize the image plane
    image_size = nxy * dxy
    x = np.linspace(-image_size/2, image_size/2, nxy)
    y = np.linspace(-image_size/2, image_size/2, nxy)
    xx, yy = np.meshgrid(x, y)

    xinc = xx/np.cos(inc)
    radius_vec = np.hypot(xinc, yy)

    ring_model = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
    
    return ring_model

def twod_gaussring(pars, args, vis_data):

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec = pars
    start, step, numsteps, nxy, dxy = args
    u, v, re, im, w = vis_data

    # convert to radians
    sigma *= arcsec
    ring_rad *= arcsec

    inclination *= deg
    posangle *= deg

    dRA *= arcsec
    dDec *= arcsec

    model_img = gaussring_model(peak, sigma, ring_rad, inclination, nxy, dxy) 
    chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA, dDec=dDec, PA=posangle, origin='lower')

    return chi2

#########################
### 2D Gaussian Ring + 1 blob
#########################

def gaussblob(peak, sigma, xx, yy, xoff, yoff, dxy):
    return 10**peak * np.exp((-1/2) * (((xx-xoff)/sigma)**2 + ((yy-yoff)/sigma)**2)) * (dxy**2)

def gaussring_1blob_model(peak, sigma, rad, inc, peak_b, sigma_b, dist, ang, nxy, dxy):
    #Initialize the image plane
    image_size = nxy * dxy
    x = np.linspace(-image_size/2, image_size/2, nxy)
    y = np.linspace(-image_size/2, image_size/2, nxy)
    xx, yy = np.meshgrid(x, y)

    #ring
    xinc = xx/np.cos(inc)
    radius_vec = np.hypot(xinc, yy)

    #blob
    xdot = dist * np.sin(ang) * np.cos(inc)
    ydot = dist * np.cos(ang)

    ring_model = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
    blob_model = gaussblob(peak_b, sigma_b, xx, yy, xdot, ydot, dxy)
    
    return ring_model + blob_model

def twod_gaussring_1blob(pars, args, vis_data):

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b, sigma_b, dist, ang = pars
    start, step, numsteps, nxy, dxy = args
    u, v, re, im, w = vis_data

    # convert to radians
    sigma *= arcsec
    ring_rad *= arcsec

    inclination *= deg
    posangle *= deg

    dRA *= arcsec
    dDec *= arcsec

    sigma_b *= arcsec
    dist *= arcsec

    ang *= deg

    model_img = gaussring_1blob_model(peak, sigma, ring_rad, inclination, peak_b, sigma_b, dist, ang, nxy, dxy) 
    chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA, dDec=dDec, PA=posangle, origin='lower')

    return chi2

#########################
### Fixed 2D Gaussian Ring + 1 blob
#########################

def twod_fixring_1blob(pars, args, vis_data):

    peak_b, sigma_b, dist, ang = pars
    start, step, numsteps, nxy, dxy = args
    u, v, re, im, w = vis_data

    # convert to radians
    sigma = 0.94 * arcsec
    ring_rad = 6.97 * arcsec

    inclination = 61.98 * deg
    posangle = 13.84 * deg

    dRA = 0.29 * arcsec
    dDec = 0.38 * arcsec

    sigma_b *= arcsec
    dist *= arcsec

    ang *= deg

    model_img = gaussring_1blob_model(6.54, sigma, ring_rad, inclination, peak_b, sigma_b, dist, ang, nxy, dxy) 
    chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA, dDec=dDec, PA=posangle, origin='lower')

    return chi2

#########################
### 2D Gaussian Ring + 2 blob in NE region
#########################

def gaussring_2blob_model(peak, sigma, rad, inc, peak_b1, sigma_b1, dist1, ang1, peak_b2, sigma_b2, dist2, ang2, nxy, dxy):
    #Initialize the image plane
    image_size = nxy * dxy
    x = np.linspace(-image_size/2, image_size/2, nxy)
    y = np.linspace(-image_size/2, image_size/2, nxy)
    xx, yy = np.meshgrid(x, y)

    #ring
    xinc = xx/np.cos(inc)
    radius_vec = np.hypot(xinc, yy)

    #blob_1
    xdot1 = dist1 * np.sin(ang1) * np.cos(inc)
    ydot1 = dist1 * np.cos(ang1)

    #blob_2
    xdot2 = dist2 * np.sin(ang2) * np.cos(inc)
    ydot2 = dist2 * np.cos(ang2)

    ring_model = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
    blob1_model = gaussblob(peak_b1, sigma_b1, xx, yy, xdot1, ydot1, dxy)
    blob2_model = gaussblob(peak_b2, sigma_b2, xx, yy, xdot2, ydot2, dxy)
    
    return ring_model + blob1_model + blob2_model

def twod_gaussring_2blob(pars, args, vis_data):

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b1, sigma_b1, dist1, ang1, peak_b2, sigma_b2, dist2, ang2 = pars
    start, step, numsteps, nxy, dxy = args
    u, v, re, im, w = vis_data

    # convert to radians
    sigma *= arcsec
    ring_rad *= arcsec

    inclination *= deg
    posangle *= deg

    dRA *= arcsec
    dDec *= arcsec

    sigma_b1 *= arcsec
    dist1 *= arcsec
    ang1 *= deg

    sigma_b2 *= arcsec
    dist2 *= arcsec
    ang2 *= deg

    model_img = gaussring_2blob_model(peak, sigma, ring_rad, inclination, peak_b1, sigma_b1, dist1, ang1, peak_b2, sigma_b2, dist2, ang2, nxy, dxy)
    chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA, dDec=dDec, PA=posangle, origin='lower')

    return chi2

###

def model_prof(pars, args, vis_data, fittype):

    """
    Routes the parameter evaluation to the appropriate physical model profile.

    Args:
        pars (np.ndarray): 1D array of model parameters to be sampled.
        args (tuple): Fixed data and constants required for the model.
        fittype (str): The model configuration identifier.

    Returns:
        float: The chi-squared value for the selected model.
    """

    if fittype == 'gaussring':
        chi2 = radial_gaussian_ring(pars, args, vis_data)

    elif fittype == 'twodgaussring':
        chi2 = twod_gaussring(pars, args, vis_data)

    elif fittype == 'twodgaussring_blob':
        chi2 = twod_gaussring_1blob(pars, args, vis_data)

    elif fittype == 'fixring_blob':
        chi2 = twod_fixring_1blob(pars, args, vis_data)

    elif fittype == 'twodring_2blob_ne':
        chi2 = twod_gaussring_2blob(pars, args, vis_data)

    else:
        logging.warning('Please choose a valid fitting model, or add a new one into the code.')

    return chi2

def model_init(fittype):

    """
    Provides starting positions and prior boundaries for a specified model.

    Args:
        fittype (str): The model configuration identifier.

    Returns:
        tuple: A 2-element tuple containing:
            - init_guess (list[float]): The initial guess starting values for walkers.
            - model_fits_ranges (list[list[float]]): The [min, max] prior bounds.
    """

    if fittype == 'gaussring':
        model_fits_initial_guesses = [6, 1, 7, 60, 15, 0, 0]
        model_fits_ranges = [[-5, 15], [0, 10], [0, 20], [0, 90], [0, 180],[-5, 5], [-5, 5]]
        logging.info('peak, sigma, ring_rad, inclination, posangle, dRA, dDec = pars')
        logging.info('start, step, numsteps, nxy, dxy = args')
        logging.info('u, v, re, im, w = vis_data')

    elif fittype == 'twodgaussring':
        model_fits_initial_guesses = [6, 1, 7, 60, 15, 0, 0]
        model_fits_ranges = [[-5, 15], [0, 10], [0, 20], [0, 90], [0, 180],[-5, 5], [-5, 5]]
        logging.info('peak, sigma, ring_rad, inclination, posangle, dRA, dDec = pars')
        logging.info('start, step, numsteps, nxy, dxy = args')
        logging.info('u, v, re, im, w = vis_data')

    elif fittype == 'twodgaussring_blob':
        model_fits_initial_guesses = [6, 1, 7, 60, 15, 0, 0, 6, 1, 7, 15]
        model_fits_ranges = [[-5, 15], [0, 10], [0, 20], [0, 90], [0, 180],[-5, 5], [-5, 5], [-5, 15], [0, 10], [0, 20], [0,90]]
        logging.info('peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b, sigma_b, distance, angle = pars')
        logging.info('start, step, numsteps, nxy, dxy = args')
        logging.info('u, v, re, im, w = vis_data')

    elif fittype == 'fixring_blob':
        model_fits_initial_guesses = [6, 1, 7, 15]
        model_fits_ranges = [[-5, 15], [0, 10], [0, 20], [0,90]]
        logging.info('peak_b, sigma_b, distance, angle = pars')
        logging.info('start, step, numsteps, nxy, dxy = args')
        logging.info('u, v, re, im, w = vis_data')

    elif fittype == 'twodring_2blob_ne':
        model_fits_initial_guesses = [6, 1, 7, 60, 15, 0, 0, 6, 1, 8, 15, 6, 1, 7, 20]
        model_fits_ranges = [[-5, 15], [0, 10], [0, 20], [0, 90], [0, 180],[-5, 5], [-5, 5], [5, 10], [0, 2], [5, 10], [0,90], [5, 10], [0, 2], [5, 10], [0,90]]
        logging.info('peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b1, sigma_b1, distance1, angle1, peak_b2, sigma_b2, distance2, angle2 = pars')
        logging.info('start, step, numsteps, nxy, dxy = args')
        logging.info('u, v, re, im, w = vis_data')

    else:
        logging.warning('Please choose a valid fitting model, or add a new one into the code.')

    return model_fits_initial_guesses, model_fits_ranges

def model_label(fittype):

    if fittype=='gaussring':
        #label = ["Peak", "$\sigma$", r"R$_{ring}$", "Inc", "PA", r"$\Delta$RA", r"$\Delta$Dec"]
        label = ["Peak", "Width", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec"]
    elif fittype=='twodgaussring':
        #label = ["Peak", "$\sigma$", r"R$_{ring}$", "Inc", "PA", r"$\Delta$RA", r"$\Delta$Dec"]
        label = ["Peak", "Width", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec"]
    elif fittype=='twodgaussring_blob':
        #label = ["Peak", "Width", "Offset", "Inc", "PA", r"$\Delta$RA", r"$\Delta$Dec"]
        label = ["Peak", "Width", "Offset", "Inc", "PA", "Offset RA", "Offset Dec", "B. Peak", "B. Width", "Dist", "Angle"]
    elif fittype=='fixring_blob':
        #label = ["Peak", "Width", "Offset", "Inc", "PA", r"$\Delta$RA", r"$\Delta$Dec"]
        label = ["B. Peak", "B. Width", "Dist", "Angle"]
    elif fittype=='twodring_2blob_ne':
        #label = ["Peak", "Width", "Offset", "Inc", "PA", r"$\Delta$RA", r"$\Delta$Dec"]
        label = ["Peak", "Width", "Offset", "Inc", "PA", "Offset RA", "Offset Dec", "B1. Peak", "B1. Width", "Dist1", "Angle1", "B2. Peak", "B2. Width", "Dist2", "Angle2"]
    else:
        logging.warning('Please choose a valid fitting model, or add a new one into the code.')
    
    return label