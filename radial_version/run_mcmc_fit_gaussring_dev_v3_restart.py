import multiprocessing

# 1. THE MASTER KEY: Force Fork before anything else happens
try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass

import os
import sys
import ast
import time
import corner
import logging
import argparse
import psutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import emcee
from scipy.special import j1
from astropy import units as u
from multiprocessing import Pool
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from galario.double import get_image_size, chi2Profile, deg, arcsec, chi2Image

from model_profiles_v3 import model_init, model_prof

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

def log_prior(pars, ranges):
    """
    Calculates the flat log-prior probability for the current parameter set.

    Args:
        pars (np.ndarray): 1D vector of model parameters to be sampled.
        ranges (np.ndarray): 2D vector of boundary constraints for the prior.

    Returns:
        float: 0.0 if all parameters are within bounds, -np.inf otherwise.
    """

    if np.any((pars <= ranges[:, 0]) | (pars >= ranges[:, 1])):
        return -np.inf
    return 0.0

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

    chi2 = model_prof(pars, args, vis_data, fittype)

    return -0.5 * chi2

def log_probability(theta, args, fittype, ranges):
    """
    Calculates the total log-posterior probability.

    Combines the prior and likelihood. If the prior returns -np.inf, 
    the likelihood is skipped to optimize computation.

    Args:
        theta (np.ndarray): 1D vector (pars) of model parameters being sampled.
        args (tuple): Fixed data and constants required for the model.
        fittype (str): The model configuration identifier.
        ranges (np.ndarray): 2D vector of boundary constraints for the prior.

    Returns:
        float: Total log-probability. Returns -np.inf if theta is out of bounds.
    """

    lp = log_prior(theta, ranges)

    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + log_likelihood(theta, args, fittype)

def initialize_data(param_file):

    """
    Parses configuration and prepares fixed observational data for the model.

    Args:
        param_file (str): Path to the parameter text file, corresponding to fittype.

    Returns:
        tuple: Bundle of fixed arguments in is a specific order, corresponding to fittype. (currenty configured for gaussring)
    """

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

    #logging.info(f"Parameter File Contents: {params}")

    u, v, re, im, w = np.require(np.loadtxt(params['visFile'], unpack=True), requirements='C')
    wavelength = 299792458/params['obsFreq']
    u /= wavelength
    v /= wavelength

    nx, dx = get_image_size(u, v)

    args = (params['radiusStart'], params['radiusStep'], params['radiusNumSteps'], nx, dx)
    vis_data = (u, v, re, im, w)

    return args, vis_data

def log_resource_usage(label=""):
    parent = psutil.Process(os.getpid())
    
    # Start with the parent's memory
    total_mem = parent.memory_info().rss
    
    # Add the memory of all 8 worker children
    for child in parent.children(recursive=True):
        try:
            total_mem += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass # Handle processes closing while we are measuring
            
    mem_mb = total_mem / (1024 * 1024)
    cpu_pct = psutil.cpu_percent(interval=None)
    num_children = len(parent.children(recursive=True))
    
    logging.info(f"RESOURCE USAGE [{label}] - CPU: {cpu_pct}% | RAM: {mem_mb:.2f} MB | Active Workers: {num_children}")

def save_results_hdf(fittype, outname, chainname):

    reader = emcee.backends.HDFBackend(chainname)
    samples = reader.get_chain()
    nsteps, nwalkers, ndim = samples.shape

    logging.info('Read in hdf5 chain file...')
    logging.info("Samples shape: %d steps, %d walkers, %d dims", nsteps, nwalkers, ndim)

    if fittype=='gaussring':
        #label = ["Peak", "$\sigma$", r"R$_{ring}$", "Inc", "PA", r"$\Delta$RA", r"$\Delta$Dec"]
        label = ["Peak", "Width", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec"]
    elif fittype=='twodgaussring':
        #label = ["Peak", "$\sigma$", r"R$_{ring}$", "Inc", "PA", r"$\Delta$RA", r"$\Delta$Dec"]
        label = ["Peak", "Width", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec"]
    elif fittype=='jinc':
        #label = ["Peak", "Width", "Offset", "Inc", "PA", r"$\Delta$RA", r"$\Delta$Dec"]
        label = ["Peak", "Width", "Offset", "Inc", "PA", "Offset RA", "Offset Dec"]
    else:
        logging.warning('Please choose a valid fitting model, or add a new one into the code.')
        return None

    if len(label) != ndim:
        logging.error(f"Dimension mismatch! File has {ndim} params, but labels has {len(label)}")
        return None

    #paths plot
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, nsteps)
        ax.set_ylabel(label[i])

    axes[0].set_title(fittype+' paths')
    axes[-1].set_xlabel("step number")
    fig.tight_layout()
    fig.savefig(outname+'_paths.png')
    logging.info('Successfully saved the paths plot...')

    #corner plot
    flat_samples = reader.get_chain(discard=200, flat=True)
    try:
        fig = corner.corner(flat_samples, labels=label,
                    show_titles=True, quantiles=[0.16, 0.50, 0.84],
                    label_kwargs={'labelpad':20, 'fontsize':0}, fontsize=8)
        fig.savefig(outname+'_corner.png')
        logging.info('Successfully saved the corner plot...')
    except:
        #Most likely a TeX error, so we want the extra info
        logging.error('Something went wrong in the creation of the corner plot. Moving on...', exc_info=True)
#eventually this will be moved to its own plot visualization script

PARAM_FILE = "/arc/home/pknowlton/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK/radial_version/gaussring_fitting_params.txt" 
args, GLOBAL_DATA = initialize_data(PARAM_FILE)
#global variable where we will store our visibility data, this should help the code run faster

def main():
    """
    Main execution script for the MCMC fitting process.

    Performs the following steps:
    1. Sets up logging and parses CLI arguments for fittype and paths.
    2. Calls model_init and initialize_data to prepare the environment.
    3. Initializes walkers within the prior-valid parameter space.
    4. Runs emcee.EnsembleSampler with multiprocessing.
    5. Monitors convergence using autocorrelation time.
    6. Saves the resulting Markov chains to an HDF5 backend.
    """

    parser=argparse.ArgumentParser()
    parser.add_argument("-op", "--outpath", default="", type=str, help="Path to product directory, corresponds with CANFAR name")
    parser.add_argument("fittype", choices=["gaussring", "twodgaussring"], default="gaussring", type=str, help="Model as specified in model_profiles.py")
    parser.add_argument("-fp", "--file_path", default="", type=str, help="Path to param file if it is not in the same directory")
    pargs=parser.parse_args()

    outpath = pargs.outpath
    fittype = pargs.fittype
    filepath = pargs.file_path

    #set up logger and psutil tracker
    logging.basicConfig(filename=outpath+'.log', filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)
    psutil.cpu_percent(interval=None)

    run_name = outpath.split('/')[-1]

    logging.info('Current Run: %s', run_name)
    log_resource_usage("Baseline")
    logging.info('Pulling arguments from argparse...')
    logging.info('Fittype: %s', fittype)

    param_file = os.path.join(filepath, fittype + '_fitting_params.txt')
    logging.info('Parameter File: %s', param_file)

    logging.info('Output Files: %s', outpath)

    #get the args
    #args, vis_data = initialize_data(param_file)
    logging.info('Initializing parameter file and data...(except not this time)')

    #log_resource_usage("Initialize Data")

    #get ranges and guesses
    init_guess, ranges = model_init(fittype)
    init_guess = np.array(init_guess)
    ranges = np.array(ranges)
    logging.info('Definiting initial guesses and prior ranges...')
    log_resource_usage("Initial Guesses and Ranges")

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
    logging.info('All guesses are within prior range...')
    log_resource_usage("Walker Setup")

    chain = outpath+'_chain.hdf5'
    backend = emcee.backends.HDFBackend(chain)

    if os.path.exists(chain) and backend.iteration > 0:
        # Resume mode
        pos = backend.get_last_sample().coords 
        logging.info(f"Resuming from iteration {backend.iteration}")
        logging.info(f"Last sample shape: {pos.shape}") # Should be (32, ndim)
    else:
        # Fresh start mode
        backend.reset(nwalkers, ndim)
        # 'pos' remains the array of initial guesses you generated above
        logging.info(f"Starting fresh run. Chain: {chain}")

    #backend.reset(nwalkers, ndim)
    #logging.info('Chain: %s', chain)

    max_n = 20000

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)

    # This will be useful to testing convergence
    old_tau = np.inf

    logging.info('Beginning emcee run...')
    log_resource_usage("Pre-Pool")

    with Pool(16) as pool:

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(args, fittype, ranges), backend=backend, pool=pool)

        steps_to_run = max_n - backend.iteration
        logging.info(f"Running {steps_to_run} steps to reach total of {max_n}...")
        
        logging.info('Beginning the fitting run...')
        fit_start=time.time()

        for sample in sampler.sample(pos, iterations=steps_to_run, progress=False, skip_initial_state_check=True):
            # Only check convergence every 100 steps
            if sampler.iteration % 100 != 0 and sampler.iteration != max_n:
                continue

		    # Record usage NOW while the pool is active and crunching numbers
            cpu = psutil.cpu_percent(interval=0.1) # interval=0.1 gives a real 'live' reading
            ram = psutil.Process().memory_info().rss / 1024 / 1024
            logging.info(f"LIVE RESOURCE USAGE - Iteration {sampler.iteration} - CPU: {cpu}% | RAM: {ram:.2f} MB")
            
            # Compute the autocorrelation time so far
		    # Using tol=0 means that we'll always get an estimate even
		    # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0, quiet=True, thin=10) #should make getting autocorr easier
            mean_tau = np.mean(tau)
            autocorr[index] = mean_tau
            index += 1

            conv_ratio = sampler.iteration / mean_tau if mean_tau > 0 else 0
            conv_ratio_all = sampler.iteration / tau if np.all(tau > 0) else np.zeros_like(tau)
            tau_diff = (old_tau - tau) / tau if np.all(tau > 0) else np.zeros_like(tau)

            logging.info(f"Iteration {sampler.iteration}/{max_n} | Mean Tau: {mean_tau:.2f} | Mean Conv Ratio: {conv_ratio:.1f}/50")
            ratios_str = ", ".join([f"{r:.1f}" for r in conv_ratio_all])
            logging.info(f"All Conv Ratios: [{ratios_str}]")
            diffs_str = ", ".join([f"{d:.2f}" for d in tau_diff])
            logging.info(f"Tau Stability: [{diffs_str}]")

            # Check convergence
            converged = np.all(tau * 50 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
        
        logging.info('Ending the fitting run...')
        log_resource_usage("Pool Usage")
        fit_end=time.time()
        logging.info("Duration of the fitting run is {0:.1f} seconds".format(fit_end-fit_start))

    log_resource_usage("Post-Pool")
    logging.info('Plotting results...')
    save_results_hdf(fittype, outpath, chain)
    log_resource_usage("Saving Results")

    logging.info('Done!')

if __name__=='__main__':
    main()