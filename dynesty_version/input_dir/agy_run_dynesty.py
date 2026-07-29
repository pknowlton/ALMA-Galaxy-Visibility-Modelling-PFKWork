import os

# 1. Enforce single-threading BEFORE loading C extensions
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import logging
import argparse
import psutil
import numpy as np
import multiprocessing
import dynesty
from dynesty.pool import Pool
from dynesty import utils as dyfunc

from galario.double import get_image_size
from model_prof import model_prof, model_addon
import prior_tform

###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%######%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%######%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###

GLOBAL_DATA = None

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

###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%######%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%######%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###

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
    
    chi2 = model_prof(pars, args, GLOBAL_DATA, 'chi2', fittype)

    return -0.5 * chi2

args, GLOBAL_DATA = initialize_data('uvtable.txt')
#global variable where we will store our visibility data, this should help the code run faster

def main():

    global GLOBAL_DATA

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

    labels, units, ndim = model_addon(fittype)

    logging.info(f"Dimensions for {fittype}: {ndim}")
    logging.info(f"Parameters for {fittype}: {labels}")

    check_file = './output/'+fittype+'_checkpoint.hdf5'
    #hist_file = './output/'+fittype+'_history.hdf5'
    fit_start=time.perf_counter()

    logging.info('Initializing Dynesty NestedSampler with dynesty.pool.Pool(16)...')
    with Pool(16, log_likelihood, prior_transform, logl_args=(args, fittype)) as pool:
        sampler = dynesty.NestedSampler(
            pool.loglike,
            pool.prior_transform,
            ndim,
            #save_evaluation_history=True,
            #history_filename=hist_file,
            nlive=1500,
            bound='multi',
            sample='rslice',
            slices=10,
            pool=pool
        )

        logging.info('Running Nested Sampler with dlogz=0.5...')

        sampler.run_nested(
            dlogz=0.5, maxiter=20000,
            checkpoint_file=check_file
        )

    logging.info('Sampling finished successfully. Results saved to %s', check_file)
    log_resource_usage()
    fit_end=time.perf_counter()
    wall_time = fit_end-fit_start
    logging.info("Duration of the fitting run is {0:.1f} seconds".format(wall_time))
    logging.info("Duration of the fitting run is {0:.1f} days".format((wall_time)/86400))

    logging.info("==========Dynesty Diagnostics Summary==========")

    res = sampler.results

    nlive = res.nlive
    niter = res.niter
    ncall = np.sum(res.ncall)
    eff = res.eff
    eval_thruput = ncall / wall_seconds

    logging.info(f"number of live points          : {nlive} ")
    logging.info(f"number of iterations           : {niter} ")
    logging.info(f"total number of function calls : {ncall} ")
    logging.info(f"overall sampling efficiency    : {eff:.2f}%")
    logging.info(f"function call/wall time        : {eval_thruput:.2f} evals/sec")

    logging.info("==========Dynesty Results Summary==========")

    logz = res.logz[-1]
    logz_err = res.logzerr[-1]

    # Compute remaining dlogz at stopping point (before final live points were added)
    logl_live = res.logl[-res.nlive:]
    logvol_stop = res.logvol[res.niter - 1]
    logz_stop = res.logz[res.niter - 1]
    logz_remain = np.max(logl_live) + logvol_stop
    remaining_dlogz = np.logaddexp(logz_stop, logz_remain) - logz_stop

    weights = np.exp(res.logwt - logz)

    logging.info(f"log evidence ln(Z)             : {logz:.3f} +/- {logz_err:.3f}")
    logging.info(f"remaining dlogz                : {remaining_dlogz:.4f}")

    for i in range(ndim):
    # Calculate 16th, 50th (median), and 84th percentiles for the 1-sigma interval
        quantiles = dyfunc.quantile(res.samples[:, i], [0.159, 0.5, 0.841], weights=weights)
    
        median = quantiles[1]
        minus_1sig = median - quantiles[0]
        plus_1sig = quantiles[2] - median

        logging.info(f"{labels[i]:<30} : {median:.3f}  (+{plus_1sig:.3f} / -{minus_1sig:.3f}) {units[i]}")

if __name__ == '__main__':

    # Set start method safely before pool creation
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    main()
