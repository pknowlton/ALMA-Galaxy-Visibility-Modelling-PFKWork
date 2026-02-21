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
    #peak, sigma, ring_rad, inclination, posangle, dRA, dDec = pars
    #visfile, obsfreq, start, step, numsteps, nxy, dxy, u, v, re, im, w = args

    for i in range(len(pars)):
        if not ranges[i][0] < pars[i] < ranges[i][1]:
            return -np.inf
    return 0.0

def log_likelihood(pars, args, fittype):
    #peak, sigma, ring_rad, inclination, posangle, dRA, dDec = pars
    #visfile, obsfreq, start, step, numsteps, nxy, dxy, u, v, re, im, w = args

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

###well get rid of this soon
def save_results(fittype, sampler, pos, outname): #prob, state
    """
    Save the outputs of the sampler for further analysis (given the fact that this is run as a script in a headless session) if necessary.
    Also creates the corner plot
    fittype: string, indicating which fit profile to use
    sampler: an ensembleSampler object that the fiting was run on (want to access the chain)
    pos: array, final position vector of all the walkers. Used later to find the fitted parameter values
    outname: string, indicating the base filename to save products to
    """
    np.savetxt(outname+'_pos.txt', pos, fmt='%10.6e', delimiter='\t')
    #I previously had been saving ALL the outputs, as I thought I would use them all. I did not. Uncomment and add prob, state to the args if you want to save these 
    #np.savetxt(outname+'_prob.txt', prob, fmt='%10.6e', delimiter='\t')
    #The state is a listbut each of the entries is a different type (string, numpy array) so I think I have to manually write to a file if we want to use it again later.
    #with open(outname+'_state.txt', 'w') as newfile:
    #    for entry in state:
    #        newfile.write(f"{entry}\n")
    
    ndim = pos.shape[1]
    try:
        samples_shape = sampler.chain.shape
        samples = sampler.chain.reshape((-1, ndim))
        np.savetxt(outname+'_chain.txt', samples, fmt='%10.6e', delimiter='\t', header=str(samples_shape))
    except:
        logging.warning('Saving the chain failed. Falling back to the documentation style of doing things.')
    logging.info('Successfully saved the sampler output variables to text files for later use.')

    #samples = sampler.chain[:, -1000:, :].reshape((-1, ndim))
    samples = sampler.chain.reshape((-1, ndim))

    if fittype=='gaussring':
        #label = ["Peak", "$\sigma$", r"R$_{ring}$", "Inc", "PA", r"$\Delta$RA", r"$\Delta$Dec"]
        label = ["Peak", "Width", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec"]
    elif fittype=='jinc':
        #label = ["Peak", "Width", "Offset", "Inc", "PA", r"$\Delta$RA", r"$\Delta$Dec"]
        label = ["Peak", "Width", "Offset", "Inc", "PA", "Offset RA", "Offset Dec"]
    else:
        logging.warning('Please choose a valid fitting model, or add a new one into the code.')
        return None

    try:
        fig = corner.corner(samples, labels=label,
                    show_titles=True, quantiles=[0.16, 0.50, 0.84],
                    label_kwargs={'labelpad':20, 'fontsize':0}, fontsize=8)
        fig.savefig(outname+'_corner.png')
        logging.info('Successfully saved the corner plot')
    except:
        #Most likely a TeX error, so we want the extra info
        logging.error('Something went wrong in the creation of the corner plot. Moving on...', exc_info=True)
###

def main():

    #well get rid of this soon
    productname = '/arc/home/pknowlton/uv_product_dir/test1/gaussing_test1'

    #Set up the logger
    logging.basicConfig(filename=productname+'.log', filemode='w', level=logging.INFO)

    #pass arguments
    parser=argparse.ArgumentParser()
    parser.add_argument("fittype", choices=["gaussring"], default="gaussring", type=str, help="Model as specified in model_profiles.py")
    parser.add_argument("-fp", "--file_path", default="", type=str, help="Path to param file if it is not in the same directory")
    pargs=parser.parse_args()

    fittype = pargs.fittype
    filepath = pargs.file_path
    logging.info('passing arguments from argparse')
    logging.info(fittype)
    logging.info(filepath)

    param_file = os.path.join(filepath, fittype + '_fitting_params.txt')
    logging.info(param_file)

    #get ranges and guesses
    init_guess, ranges = model_init(fittype)
    logging.info('pulling guesses and ranges')
    
    #get the args
    args = initialize_data(param_file)
    logging.info('reading param file and initializing args')

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
    logging.info('All guesses within prior range')

    #eventually set up hdf5 chain

    logging.info('begging emcee')
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(args, fittype, ranges), pool=pool)
        logging.info('Beginning the burn-in...')
        bi_start = time.time()
        pos0, _, _ = sampler.run_mcmc(pos, 50) #test, set to 3000
        sampler.reset()
        bi_end=time.time()
        logging.info("Duration of the burn-in is {0:.1f} seconds".format(bi_end-bi_start))
        logging.info('Beginning the fitting run...')
        fit_start=time.time()
        pos1, _, _ = sampler.run_mcmc(pos0, 5) #test, set to 3000
        fit_end=time.time()
        logging.info("Duration of the fitting run is {0:.1f} seconds".format(fit_end-fit_start))

    logging.info('saving results')
    save_results(fittype, sampler, pos1, productname)

    logging.info('done!')

if __name__=='__main__':
    main()