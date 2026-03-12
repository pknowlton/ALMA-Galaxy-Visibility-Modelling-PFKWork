import os
import sys
import ast
import time
import corner
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import emcee
from scipy.special import j1
from astropy import units as u
from multiprocessing import Pool
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from galario.double import get_image_size, chi2Profile, deg, arcsec

from model_profiles import model_init, model_prof

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

    for i in range(len(pars)):
        if not ranges[i][0] < pars[i] < ranges[i][1]:
            return -np.inf
    return 0.0

def log_likelihood(pars, args, fittype):

    chi2 = model_prof(pars, args, fittype)

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

def save_results_hdf(fittype, outname, chainname):

    reader = emcee.backends.HDFBackend(chainname)
    samples = reader.get_chain()
    nsteps, nwalkers, ndim = samples.shape

    logging.info('Read in hdf5 chain file...')
    logging.info("Samples shape: %d steps, %d walkers, %d dims", nsteps, nwalkers, ndim)

    if fittype=='gaussring':
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
    flat_samples = reader.get_chain(discard=0, flat=True)
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
    

def main():

    #well get rid of this soon
    productname = '/arc/home/pknowlton/uv_product_dir/test6/gaussing_test6_short_pool1'

    #Set up the logger
    logging.basicConfig(filename=productname+'.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    #pass arguments
    parser=argparse.ArgumentParser()
    parser.add_argument("fittype", choices=["gaussring"], default="gaussring", type=str, help="Model as specified in model_profiles.py")
    parser.add_argument("-fp", "--file_path", default="", type=str, help="Path to param file if it is not in the same directory")
    pargs=parser.parse_args()

    fittype = pargs.fittype
    filepath = pargs.file_path
    logging.info('Pulling arguments from argparse...')
    logging.info('Fittype: %s', fittype)

    param_file = os.path.join(filepath, fittype + '_fitting_params.txt')
    logging.info('Parameter File: %s', param_file)

    #get ranges and guesses
    init_guess, ranges = model_init(fittype)
    logging.info('Definiting initial guesses and prior ranges...')
    
    #get the args
    args = initialize_data(param_file)
    logging.info('Initializing parameter file and defining args...')

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

    chain = productname+'_chain.hdf5'
    backend = emcee.backends.HDFBackend(chain)
    backend.reset(nwalkers, ndim)
    logging.info('Chain: %s', chain)

    max_n = 15

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)

    # This will be useful to testing convergence
    old_tau = np.inf

    logging.info('Beginning emcee run...')

    with Pool(1) as pool:

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(args, fittype, ranges), backend=backend, pool=pool)
        
        logging.info('Beginning the fitting run...')
        fit_start=time.time()

        for sample in sampler.sample(pos, iterations=max_n, progress=False, skip_initial_state_check=True):
            # Only check convergence every 1 steps
            if sampler.iteration % 1:
                continue

		    # Compute the autocorrelation time so far
		    # Using tol=0 means that we'll always get an estimate even
		    # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0, quiet=True, thin=1)
            mean_tau = np.mean(tau)
            autocorr[index] = mean_tau
            index += 1

            conv_ratio = sampler.iteration / mean_tau if mean_tau > 0 else 0
            logging.info(f"Iteration {sampler.iteration}/{max_n} | Mean Tau: {mean_tau:.2f} | Conv Ratio: {conv_ratio:.1f}/100")

            # Check convergence
            converged = np.all(tau * 50 < sampler.iteration) #trying 50 for now
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
        
        logging.info('Ending the fitting run...')
        fit_end=time.time()
        logging.info("Duration of the fitting run is {0:.1f} seconds".format(fit_end-fit_start))

    logging.info('Plotting results...')
    save_results_hdf(fittype, productname, chain)

    logging.info('Done!')

if __name__=='__main__':
    main()