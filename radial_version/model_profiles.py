import numpy as np
import logging

### running list of model profiles

def radial_gaussian_ring(pars, args):

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec = pars
    start, step, numsteps, nxy, dxy, u, v, re, im, w = args

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

def model_prof(pars, args, fittype):

    if fittype == 'gaussring':
        chi2 = radial_gaussian_ring(pars, args)

    else:
        logging.warning('Please choose a valid fitting model, or add a new one into the code.')

    return chi2

def model_guess(fittype):

    if fittype == 'gaussring':
        model_fits_initial_guesses = [6, 1, 7, 60, 15, 0, 0]
        model_fits_ranges = [[-5, 15], [0, 10], [0, 20], [0, 90], [0, 180],[-5, 5], [-5, 5]]

    else:
        logging.warning('Please choose a valid fitting model, or add a new one into the code.')

    return model_fits_initial_guesses, model_fits_ranges
