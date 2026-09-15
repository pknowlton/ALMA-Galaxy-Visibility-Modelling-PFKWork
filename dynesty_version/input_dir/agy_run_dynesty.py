"""
Bayesian Visibility Modeling with Dynesty Dynamic Nested Sampling
===================================================================
This script performs Bayesian parameter estimation and model selection by fitting
2D brightness profile models directly to calibrated ALMA interferometric visibility
data $(u, v, \text{Re}, \text{Im}, \text{weight})$ using the **Dynesty** dynamic
nested sampling library and **Galario**.

Key Concepts:
- **Interferometric Visibilities**: Radio interferometers like ALMA do not take direct
  sky pictures; they sample spatial Fourier frequencies $(u, v)$ of the sky brightness.
  Fitting models directly in the Fourier $(u, v)$ domain avoids deconvolution artifacts
  and non-linear imaging biases.
- **Dynamic Nested Sampling**: Simultaneously explores multi-modal parameter spaces,
  computes the Bayesian evidence $\mathcal{Z} = \int \mathcal{L}(\theta)\pi(\theta)d\theta$
  (for model comparison), and produces posterior probability distributions with
  reliable uncertainty estimates.
- **Multiprocessing Pool**: Uses `dynesty.pool.Pool` to evaluate log-likelihoods
  in parallel across CPU cores. Threading libraries (OpenMP/MKL/OpenBLAS) are restricted
  to single threads to prevent CPU oversubscription.

Usage:
    python agy_run_dynesty.py <fittype>

Example:
    python agy_run_dynesty.py twod_gaussring
"""

import os

# 1. Enforce single-threading BEFORE loading C extensions
# Restricting backend numerical libraries to 1 thread prevents thread contention
# when multiprocessing distributes likelihood evaluations across multiple CPU cores.
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
from dynesty import DynamicNestedSampler

from galario.double import get_image_size
from model_prof import model_prof, model_addon
import prior_tform

###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%######%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%######%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###

# Module-level variable to store visibility data, avoiding inter-process serialization overhead
GLOBAL_DATA = None

def initialize_data(data_file):
    """
    Loads and prepares interferometric visibility data for Galario.

    Reads baseline coordinates $(u, v)$ in meters, observed complex visibilities
    $(\text{Re}, \text{Im})$, and statistical weights from an ASCII table.
    Converts $(u, v)$ baselines into units of observing wavelength ($\lambda = c/\nu$),
    and computes the required simulation image dimensions $(N_{xy}, \Delta xy)$ using
    Galario's Nyquist sampling criteria.

    Parameters
    ----------
    data_file : str
        Path to the ASCII visibility table (columns: u, v, Re, Im, weight).

    Returns
    -------
    args : tuple
        Image grid parameters `(nxy, dxy)`:
            - `nxy` (int): Number of pixels along image edge.
            - `dxy` (float): Pixel scale in radians.
    vis_data : tuple
        Visibility arrays `(u, v, re, im, w)`:
            - `u, v` (numpy.ndarray): Spatial frequencies in units of wavelength.
            - `re, im` (numpy.ndarray): Real and Imaginary visibility components in Jy.
            - `w` (numpy.ndarray): Statistical weights ($1/\sigma^2$) in $\text{Jy}^{-2}$.
    """
    u, v, re, im, w, freq = np.require(np.loadtxt(data_file, unpack=True), requirements='C')
    wavelength = 299792458 / 93e9  # 93 GHz continuum wavelength (~3.22 mm)
    u /= wavelength
    v /= wavelength

    # Calculate optimal image size and pixel scale to avoid aliasing in Fourier space
    nx, dx = get_image_size(u, v)

    args = (nx, dx)
    vis_data = (u, v, re, im, w)

    return args, vis_data

###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%######%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%######%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###

def log_likelihood(pars, args, fittype):
    """
    Calculates the log-likelihood probability $\ln \mathcal{L}$ under a Gaussian noise model.

    Evaluates the model visibilities via Galario and computes the total $\chi^2$ sum:
        $\ln \mathcal{L} = -\frac{1}{2} \chi^2 = -\frac{1}{2} \sum_k w_k \left| V_{\text{obs}, k} - V_{\text{mod}, k} \right|^2$

    Parameters
    ----------
    pars : numpy.ndarray
        1D vector of physical model parameters proposed by the sampler.
    args : tuple
        Image grid properties `(nxy, dxy)` passed to Galario.
    fittype : str
        Model configuration name (e.g. `'twod_gaussring'`).

    Returns
    -------
    float
        The log-likelihood value $-0.5 \cdot \chi^2$.
    """
    global GLOBAL_DATA  # Access globally cached visibility data
    
    chi2 = model_prof(pars, args, GLOBAL_DATA, 'chi2', fittype)

    return -0.5 * chi2

# Pre-load visibility dataset into global memory
args, GLOBAL_DATA = initialize_data('uvtable.txt')

def main():
    """
    Main driver for Dynesty dynamic nested sampling.

    Workflow:
    1. Parses CLI argument for model name (`fittype`).
    2. Initializes file logger in `./output/<fittype>_dynesty.log`.
    3. Dynamically resolves the corresponding prior transform function from `prior_tform.py`.
    4. Configures `dynesty.DynamicNestedSampler` with a multiprocessing pool (16 workers).
    5. Executes nested sampling until target evidence precision ($\Delta \ln \mathcal{Z} \le 0.5$) is reached.
    6. Saves checkpoint state to disk (`./output/<fittype>_checkpoint.save`).
    7. Computes and logs diagnostics (evidence $\ln \mathcal{Z}$, efficiency, throughput)
       and parameter posterior quantiles (median, $\pm 1\sigma$).
    """
    global GLOBAL_DATA

    parser = argparse.ArgumentParser(
        description="Run Dynesty dynamic nested sampling for ALMA visibility modeling."
    )
    parser.add_argument("fittype", type=str, help="Model profile identifier (e.g., 'twod_gaussring').")
    pargs = parser.parse_args()

    fittype = pargs.fittype

    # Configure run logger
    logfile = './output/' + fittype + '_dynesty.log'
    logging.basicConfig(
        filename=logfile,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )
    logging.info('Fittype: %s', fittype)

    logging.info('Pulling correct prior transform function from prior_tform.py')

    # Dynamically look up prior transform function matching the model name
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


    logging.info("==========Prior Ranges==========")
    low_u = np.zeros(ndim)
    low_v = prior_transform(low_u)
    high_u = np.zeros(ndim)+1
    high_v = prior_transform(high_u)
    
    for i in range(ndim):

        logging.info(f"{labels[i]:<30} [{low_v[i]:.1f}, {high_v[i]:.1f}] {units[i]}")

    check_file = './output/' + fittype + '_checkpoint.save'
    fit_start = time.perf_counter()

    # Initialize multiprocessing Pool and Dynamic Nested Sampler
    logging.info('Initializing Dynesty DynamicNestedSampler with dynesty.pool.Pool(16)...')
    with Pool(16, log_likelihood, prior_transform, logl_args=(args, fittype)) as pool:
        sampler = dynesty.DynamicNestedSampler(
            pool.loglike,
            pool.prior_transform,
            ndim,
            nlive=250,          # Number of live points for initial nested sampling run
            bound='multi',       # Use multi-ellipsoid bounding distributions
            sample='rwalk',      # Random-walk sampling within bounds (suitable for ndim > 5)
            pool=pool,
            queue_size=16
        )
        
        # Run sampling until the stopping criterion (dlogz_init <= 0.5) is satisfied
        sampler.run_nested(
            dlogz_init=0.5,
            maxbatch=0,          # Perform initial baseline run without extra dynamic batches
            maxiter=None,
            maxcall=None,
            checkpoint_file=check_file
        )

    logging.info('Sampling finished successfully. Results saved to %s', check_file)
    fit_end = time.perf_counter()
    wall_time = fit_end - fit_start
    logging.info("Duration of the fitting run is {0:.1f} seconds".format(wall_time))
    logging.info("Duration of the fitting run is {0:.1f} days".format((wall_time) / 86400))

    # --------------------------------------------------------------------------
    # Dynesty Diagnostics & Performance Metrics
    # --------------------------------------------------------------------------
    logging.info("==========Dynesty Diagnostics Summary==========")

    res = sampler.results

    n_mean = int(np.round(np.mean(res.samples_n)))
    niter = res.niter
    ncall = np.sum(res.ncall)
    eff = res.eff
    eval_thruput = ncall / wall_time

    logging.info(f"average number of live points  : {n_mean} ")
    logging.info(f"number of iterations           : {niter} ")
    logging.info(f"total number of function calls : {ncall} ")
    logging.info(f"overall sampling efficiency    : {eff:.2f}%")
    logging.info(f"function call/wall time        : {eval_thruput:.2f} evals/sec")

    # --------------------------------------------------------------------------
    # Dynesty Parameter Estimation & Evidence Summary
    # --------------------------------------------------------------------------
    logging.info("==========Dynesty Results Summary==========")

    logz = res.logz[-1]
    logz_err = res.logzerr[-1]

    # Compute remaining evidence fraction dlogz at termination
    logl_max = np.max(res.logl)      # Maximum log-likelihood across all samples
    logvol_stop = res.logvol[-1]     # Final remaining prior volume
    logz_stop = res.logz[-1]         # Accumulated log-evidence
    logz_remain = logl_max + logvol_stop
    remaining_dlogz = np.logaddexp(logz_stop, logz_remain) - logz_stop

    # Posterior sample statistical weights
    weights = np.exp(res.logwt - logz)

    logging.info(f"log evidence ln(Z)             : {logz:.3f} +/- {logz_err:.3f}")
    logging.info(f"remaining dlogz                : {remaining_dlogz:.4f}")

    # Compute 16th, 50th (median), and 84th percentiles for parameter credible intervals
    for i in range(ndim):
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
