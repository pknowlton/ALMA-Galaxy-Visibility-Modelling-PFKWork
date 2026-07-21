import numpy as np
import logging
from galario.double import get_image_size, chi2Profile, deg, arcsec, chi2Image
import pandas as pd

#########################
### toy model galario test
#########################

def gaussring_prof(peak_ring, sigma_ring, rad_ring, radius, dxy):
    return 10**peak_ring * np.exp((-1/2)*((radius-rad_ring)/sigma_ring)**2) * (dxy**2)

def model_ring_galtest(peak, sigma, rad, inc, nxy, dxy):

    #Initialize the image plane
    image_size = nxy * dxy
    x = np.linspace(-image_size/2, image_size/2, nxy)
    y = np.linspace(-image_size/2, image_size/2, nxy)
    xx, yy = np.meshgrid(x, y)

    #ring
    xinc = xx/np.cos(inc)
    radius_vec = np.hypot(xinc, yy)

    r_model = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
    
    return r_model

def model_ring_galtest_setup(pars, args, vis_data):

    posangle, dRA, dDec = pars
    start, step, numsteps, nxy, dxy = args
    u, v, re, im, w = vis_data

    # convert to radians
    sigma = 1 * arcsec
    ring_rad = 7 * arcsec
    inclination = 60 * deg
    
    posangle *= deg
    dRA *= arcsec
    dDec *= arcsec

    model_img = model_ring_galtest(6, sigma, ring_rad, inclination, nxy, dxy) 
    chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA, dDec=dDec, PA=posangle, origin='lower')

    return chi2

##################################################################################################################

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

    if fittype == 'galario_test':
        chi2 = model_ring_galtest_setup(pars, args, vis_data)

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

    if fittype == 'galario_test':
        model_fits_initial_guesses = [45, 2.0, 2.0]
        model_fits_ranges = [[0, 90], [0, 4], [0, 4]]
        logging.info('posangle, dRA, dDec = pars')
        logging.info('start, step, numsteps, nxy, dxy = args')
        logging.info('u, v, re, im, w = vis_data')

    else:
        logging.warning('Please choose a valid fitting model, or add a new one into the code.')

    return model_fits_initial_guesses, model_fits_ranges