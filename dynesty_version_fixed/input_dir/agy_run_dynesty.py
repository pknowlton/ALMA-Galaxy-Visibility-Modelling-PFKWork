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
- **Wavelength Ingestion**: Assumes external data export provides $(u, v)$ coordinates
  in units of observing wavelength per datum/channel ($u_\lambda = u / \lambda$, $v_\lambda = v / \lambda$),
  or provides an explicit frequency column, avoiding erroneous single-wavelength approximations.
- **Multiprocessing Pool**: Uses `dynesty.pool.Pool` to evaluate log-likelihoods
  in parallel across CPU cores. Threading libraries (OpenMP/MKL/OpenBLAS) are restricted
  to single threads to prevent CPU oversubscription.

Usage:
    python agy_run_dynesty.py <fittype> [--nlive 500] [--maxbatch 5] [--dlogz 0.1] [--seed 42] [--resume]

Example:
    python agy_run_dynesty.py twod_gaussring --nlive 500 --maxbatch 5 --dlogz 0.1 --seed 42
"""

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
from dynesty import DynamicNestedSampler

from galario.double import get_image_size
from model_prof import model_prof, model_addon
import prior_tform_streamline as prior_tform

# Module-level variable to store visibility data, avoiding inter-process serialization overhead
GLOBAL_DATA = None

def initialize_data(data_file):
    """
    Loads and prepares interferometric visibility data for Galario.

    Reads baseline coordinates $(u, v)$, complex visibilities $(\text{Re}, \text{Im})$,
    and statistical weights. Supports data files where $(u, v)$ are already in units
    of observing wavelength ($\lambda$), or files containing a 6th frequency column.

    Parameters
    ----------
    data_file : str
        Path to the ASCII visibility table.

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
    raw_data = np.require(np.loadtxt(data_file, unpack=True), requirements='C')

    # If table has 6+ columns, column 5 is frequency in Hz
    if len(raw_data) >= 6:
        u_raw, v_raw, re, im, w, freq = raw_data[:6]
        c_light = 299792458.0
        wavelength = c_light / freq
        u = u_raw / wavelength
        v = v_raw / wavelength
    else:
        # Assumes external export script provided (u, v) in units of wavelength (u_lambda, v_lambda)
        u, v, re, im, w = raw_data[:5]

    # Calculate optimal image size and pixel scale from baseline extrema
    nx, dx = get_image_size(u, v)

    args = (nx, dx)
    vis_data = (u, v, re, im, w)

    return args, vis_data

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
    """
    global GLOBAL_DATA

    parser = argparse.ArgumentParser(
        description="Run Dynesty dynamic nested sampling for ALMA visibility modeling."
    )
    parser.add_argument("fittype", type=str, help="Model profile identifier (e.g., 'twod_gaussring').")
    parser.add_argument("--nlive", type=int, default=250, help="Initial number of live points (default: 500).")
    parser.add_argument("--maxbatch", type=int, default=5, help="Maximum number of dynamic batches to add (default: 5).")
    parser.add_argument("--dlogz", type=float, default=0.1, help="Target evidence stopping criterion dlogz (default: 0.1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint file if it exists.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not resume from existing checkpoint.")
    pargs = parser.parse_args()

    fittype = pargs.fittype
    nlive = pargs.nlive
    maxbatch = pargs.maxbatch
    dlogz = pargs.dlogz
    seed = pargs.seed

    # Set random seed
    np.random.seed(seed)

    # Configure run logger
    os.makedirs('./output', exist_ok=True)
    logfile = './output/' + fittype + '_dynesty.log'
    logging.basicConfig(
        filename=logfile,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )
    logging.info('=================================================================')
    logging.info('Dynesty Bayesian Visibility Modeling Execution Started')
    logging.info('=================================================================')
    logging.info('Fittype: %s', fittype)
    logging.info('Random seed: %d', seed)
    logging.info('nlive_init: %d, maxbatch: %d, dlogz_init: %.3f', nlive, maxbatch, dlogz)

    # Dynamically look up prior transform function matching the model name
    try:
        prior_transform = getattr(prior_tform, f"{fittype}_ptform")
    except AttributeError:
        msg = f"No prior function '{fittype}_ptform' found in prior_tform_streamline.py, please check spelling or add function."
        logging.warning(msg)
        raise ValueError(msg)

    logging.info(f'Prior transform function is {fittype}_ptform')
    
    labels, units, ndim = model_addon(fittype)

    logging.info(f"Dimensions for {fittype}: {ndim}")
    logging.info(f"Parameters for {fittype}: {labels}")

    logging.info("==========Prior Ranges==========")
    low_u = np.zeros(ndim)
    low_v = prior_transform(low_u)
    high_u = np.zeros(ndim) + 1.0
    high_v = prior_transform(high_u)
    
    for i in range(ndim):
        logging.info(f"{labels[i]:<30} [{low_v[i]:.2f}, {high_v[i]:.2f}] {units[i]}")

    # Identify periodic dimensions (e.g. Ring PA is index 4; Angle 2 in double pendulum is index 14; simgauss PA is restricted to [0, 15] deg so not periodic)
    periodic_dims = []
    if fittype not in ('simgauss', 'twod_simgauss'):
        periodic_dims.append(4)
    if fittype == 'twod_gauss1blob_2peak_dp':
        periodic_dims.append(14)
    periodic = periodic_dims if len(periodic_dims) > 0 else None
    logging.info("Periodic parameter indices: %s", periodic_dims)

    check_file = './output/' + fittype + '_checkpoint.save'
    fit_start = time.perf_counter()

    logging.info('Initializing Dynesty DynamicNestedSampler with dynesty.pool.Pool(16)...')
    with Pool(16, log_likelihood, prior_transform, logl_args=(args, fittype)) as pool:
        if pargs.resume and os.path.exists(check_file):
            logging.info("Restoring existing sampler state from checkpoint: %s", check_file)
            try:
                sampler = DynamicNestedSampler.restore(check_file, pool=pool)
                sampler.run_nested(
                    resume=True,
                    dlogz_init=dlogz,
                    maxbatch=maxbatch,
                    checkpoint_file=check_file
                )
            except Exception as e:
                logging.warning(f"Could not resume from checkpoint ({e}). Starting fresh sampler.")
                sampler = DynamicNestedSampler(
                    pool.loglike,
                    pool.prior_transform,
                    ndim,
                    nlive=nlive,
                    bound='multi',
                    sample='rwalk',
                    periodic=periodic,
                    pool=pool,
                    queue_size=16
                )
                sampler.run_nested(
                    dlogz_init=dlogz,
                    maxbatch=maxbatch,
                    checkpoint_file=check_file
                )
        else:
            sampler = DynamicNestedSampler(
                pool.loglike,
                pool.prior_transform,
                ndim,
                nlive=nlive,
                bound='multi',
                sample='rwalk',
                periodic=periodic,
                pool=pool,
                queue_size=16
            )
            sampler.run_nested(
                dlogz_init=dlogz,
                maxbatch=maxbatch,
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
    eval_thruput = ncall / wall_time if wall_time > 0 else 0

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

    logl_max = np.max(res.logl)
    logvol_stop = res.logvol[-1]
    logz_stop = res.logz[-1]
    logz_remain = logl_max + logvol_stop
    remaining_dlogz = np.logaddexp(logz_stop, logz_remain) - logz_stop

    weights = np.exp(res.logwt - logz)

    logging.info(f"log evidence ln(Z)             : {logz:.3f} +/- {logz_err:.3f}")
    logging.info(f"remaining dlogz                : {remaining_dlogz:.4f}")

    # Maximum likelihood (best-fit) parameter vector
    best_idx = np.argmax(res.logl)
    ml_pars = res.samples[best_idx]
    logging.info("==========Maximum Likelihood (ML) Parameter Estimates==========")
    for i in range(ndim):
        logging.info(f"{labels[i]:<30} (ML) : {ml_pars[i]:.4f} {units[i]}")

    logging.info("==========Posterior Marginal Quantiles (16th, 50th, 84th)==========")
    for i in range(ndim):
        quantiles = dyfunc.quantile(res.samples[:, i], [0.159, 0.5, 0.841], weights=weights)
        median = quantiles[1]
        minus_1sig = median - quantiles[0]
        plus_1sig = quantiles[2] - median
        logging.info(f"{labels[i]:<30} : {median:.3f}  (+{plus_1sig:.3f} / -{minus_1sig:.3f}) {units[i]}")

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    main()
