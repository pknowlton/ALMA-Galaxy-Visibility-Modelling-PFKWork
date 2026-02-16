import os
import sys
import ast
import time
import corner
import logging
import argparse
import numpy as np
import pandas as pd
import emcee
from scipy.special import j1
from astropy import units as u
from multiprocessing import Pool
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from galario.double import get_image_size, chi2Profile, deg, arcsec

#import models
#import setup

def log_prior(pars, ranges):
    #peak, sigma, ring_rad, inclination, posangle, dRA, dDec = pars
    #visfile, obsfreq, start, step, numsteps, nxy, dxy, u, v, re, im, w = args

    for i in range(len(pars)):
        if not model_fits_ranges[i][0] < pars[i] < model_fits_ranges[i][1]:
            return -np.inf
    return 0.0

def log_likelihood(pars, args, fittype):
    #peak, sigma, ring_rad, inclination, posangle, dRA, dDec = pars
    #visfile, obsfreq, start, step, numsteps, nxy, dxy, u, v, re, im, w = args

    chi = model_prof(pars, args, fittype)

    return -0.5 * chi2

def log_probability(theta, args, fittype, ranges):

    lp = log_prior(theta, ranges)

    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + log_likelihood(theta, args, fittype)

def initialize_data(param_file):

    params={}
    with open(param_file, 'r') as f:
        for line in f:
            line = line.split("#", 1)[0].strip() #This will skip in-line comments
            if "=" in line:
                pkey, pvalue = line.split("=", maxsplit=1)
                pkey = pkey.strip()
                pvalue = ast.literal_eval(pvalue.strip())
                if isinstance(pvalue, list):
                    pvalue = np.array(pvalue, dtype=float)
                params[pkey] = pvalue

    u, v, re, im, w = np.require(np.loadtxt(params['visFile'], unpack=True), requirements='C')
    wavelength = 299792458/params['obsFreq']
    u /= wavelength
    v /= wavelength

    nx, dx = get_image_size(u, v)

    return params['radiusStart'], params['radiusStep'], params['radiusNumSteps'], nx, dx, u, v, re, im, w

def main():

    #what arguments needs to be passed this way?

    #get the args

    args = initialize_data(param_file)

    #get ranges and guesses

    init_guess, ranges = model_guess(fittype)

    #set up walkers
    nwalkers = 32
    ndim = len(ranges)
    pos = np.zeros([nwalkers, ndim])

    for i in range(nwalkers):
        c = 0
        while c != 1:
            pos[i, :] = init_guess + 1e-5*np.random.randn(ndim)
            chk_pars = log_prior(pos[i,:], ranges)
            if chk_pars == 0:
                c = 1
    print('All guesses within prior range')

    #peak, sigma, ring_rad, inclination, posangle, dRA, dDec = pars
    #start, step, numsteps, nxy, dxy, u, v, re, im, w = args

    #eventually set up hdf5 chain

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(args, fittype, ranges), pool=Pool(8)) #backend