import os
import sys
import time
import logging
import argparse
import psutil
import numpy as np
import multiprocessing
from dynesty.pool import Pool
import dynesty

from galario.double import get_image_size
from model_prof import model_prof, model_dim
import prior_tform

# Force fork method for multiprocessing consistency across environments
try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%######%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%######%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###

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
    vis_data = GLOBAL_DATA

    chi2 = model_prof(pars, args, vis_data, 'chi2', fittype)

    return -0.5 * chi2


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

    check_file = './output/'+fittype+'_checkpoint.hdf5'
    hist_file = './output/'+fittype+'_history.hdf5'
    fit_start=time.time()

    logging.info('Initializing Dynesty NestedSampler with dynesty.pool.Pool(16)...')
    with Pool(16, log_likelihood, prior_transform, logl_args=(args, fittype)) as pool:
        sampler = dynesty.NestedSampler(
            pool.loglikelihood,
            pool.prior_transform,
            ndim,
            nlive=250,
            bound='multi',
            sample='unif',
            pool=pool,
            queue_size=16
        )

        logging.info('Running Nested Sampler with dlogz=0.5...')

        sampler.run_nested(
            dlogz=0.5, maxiter=20000,
            checkpoint_file=check_file,
            save_evaluation_history=True,
            history_filename=hist_file,
        )

    logging.info('Sampling finished successfully. Results saved to %s', check_file)
    log_resource_usage()
    fit_end=time.time()
    logging.info("Duration of the fitting run is {0:.1f} seconds".format(fit_end-fit_start))
    logging.info("Duration of the fitting run is {0:.1f} days".format((fit_end-fit_start)/86400))


if __name__ == '__main__':
    main()
