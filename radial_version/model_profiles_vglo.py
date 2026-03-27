import numpy as np
import logging
from galario.double import get_image_size, chi2Profile, deg, arcsec

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
        logging.info('start, step, numsteps, nxy, dxy, u, v, re, im, w = args')

    else:
        logging.warning('Please choose a valid fitting model, or add a new one into the code.')

    return model_fits_initial_guesses, model_fits_ranges
