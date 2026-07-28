import multiprocessing

# 1. THE MASTER KEY: Force Fork before anything else happens
try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import sys
import ast
import time
import corner
import logging
import argparse
import psutil
import matplotlib.pyplot as plt
import pandas as pd
import dynesty
from scipy.special import j1
from astropy import units as u
from multiprocessing import Pool
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from galario.double import get_image_size, chi2Profile, deg, arcsec, chi2Image

from model_prof import model_prof, model_dim
import prior_tform

#Matplotlib used by corner, but TeX issue is causing crashes. Force usetex=False
from matplotlib import pyplot as plt
from matplotlib import rc
#rc('text', usetex=True)
rc('text', usetex=False)
font = {'family' : 'serif',
        'weight' : 'regular',
        'size'   : '14'}
rc('font', **font)
###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###

###%%%###%%%### Necessary to stop numpy from tripping over itself during parallel processing? ###%%%###%%%###
os.environ["OMP_NUM_THREADS"] = "1"
###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###

def log_likelihood(pars, args, fittype):
    """
    Calculates the log-likelihood probability, assuming a Gaussian likelihood function through the Method of Least Sqaures.

    Interface with model profile function to generate model and chi-sqaure.

    Args:
        pars (np.ndarray): 1D vector of model parameters to be sampled.
        args (tuple): Fixed data and constants required for the model.
        fittype (str): The model configuration identifier.

    Returns:
        float: The log-likelihood value, calculated as -0.5 * chi_squared.
    """

    global GLOBAL_DATA # Add this line to be safe
    vis_data = GLOBAL_DATA

    chi2 = model_prof(pars, args, vis_data, 'chi2', fittype)

    return -0.5 * chi2

def initialize_data(data_file):

    u, v, re, im, w = np.require(np.loadtxt(data_file, unpack=True), requirements='C')
    wavelength = 299792458/93e9
    u /= wavelength
    v /= wavelength

    nx, dx = get_image_size(u, v)

    args = (nx, dx)
    vis_data = (u, v, re, im, w)

    return args, vis_data

def log_resource_usage():
    parent = psutil.Process(os.getpid())
    num_children = len(parent.children(recursive=True))
    
    logging.info(f"Active Workers: {num_children}")


args, GLOBAL_DATA = initialize_data('uvtable.txt')
#global variable where we will store our visibility data, this should help the code run faster

def main():

    parser=argparse.ArgumentParser()
    parser.add_argument("fittype", type=str, help="Model as specified in model_prof.py")
    pargs=parser.parse_args()

    fittype = pargs.fittype

    #set up logger and psutil tracker
    logfile = './output/'+fittype+'_dynesty.log'
    logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)
    logging.info('Fittype: %s', fittype)

    logging.info('Pulling correct prior transtorm function from prior_tform.py')

    try:
        prior_transform = getattr(prior_tform, f"{fittype}_ptform")
    except AttributeError:
        msg = f"No prior function '{fittype}_ptform' found in prior_tform.py, please check spelling or add function."
        logging.warning(msg)
        raise ValueError(msg)

    logging.info(f'Prior transform function is {fittype}_ptform')

    ndim = model_dim(fittype)

    logging.info(f"Dimensions for {fittype}: {ndim}")







