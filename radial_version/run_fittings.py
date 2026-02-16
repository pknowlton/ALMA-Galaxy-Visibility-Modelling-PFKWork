#Some things are commented out here, as they seemed helpful at some points, but either are no longer necessary, or cause problems.
#If you have a proper TeX installation: fix lines 28, 188, 191. 
###%%%###%%%###%%%###%%%###%%%###

###%%%###%%%### Package Imports ###%%%###%%%###
import os
import sys
import ast
import time
import corner
import logging
import numpy as np
import pandas as pd
from scipy.special import j1
from astropy import units as u
from multiprocessing import Pool
from astropy.units import Quantity #Note: when I went through adding more comments to the functions, I realized this might be redundant and left over from previous troubleshooting. Unable to remove and test to check that though..
from emcee import EnsembleSampler as es
from astropy.coordinates import SkyCoord
from galario.double import get_image_size, chi2Profile

#Import the visualize_main function from the visualizeModel.py file in order to make output plots for each fit
from visualizeModel import visualize_main

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

###%%%###%%%### Fitting profiles ###%%%###%%%###  
def radial_gaussian_ring(peak, sigma, ring_rad, start, step, numsteps):
    """
    Make a 1d profile of a ring with a gaussian shape, in the formatting required for use in sampleProfile and chi2Profile.
    peak, sigma: peak and width of the gaussian
    ring_rad: radial position of the gaussian centre
    start, step, numsteps: origin, grid spacing, grid size of the radial grid to evaluate the function on
    """
    radius = np.linspace(start, start + numsteps*step, numsteps)
    return peak*np.exp((-1/2)*((radius-ring_rad)/sigma)**2)

def radial_jinc(normalization, width, offset, start, step, numsteps):
    """
    Make a 1d profile of a jinc function, in the formatting required for use in sampleProfile and chi2Profile.
    normalization, width: peak and width of the jinc function
    offset: a 'vertical' offset that can be used to keep the brightness function nonzero
    start, step, numsteps: origin (if =0 will can cause divide by zero errors), grid spacing, grid size of the radial grid to evaluate the function on
    """
    radius = np.linspace(start, start + numsteps*step, numsteps)
    return normalization*j1(width*radius)/radius + offset
###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###

###%%%###%%%### Emcee evaluation functions ###%%%###%%%###
def lnpriorfn(p, par_ranges):
    """
    Check if the parameters are within the defined range (check if meeting prior condition)
    p: list containing the parameter vector
    par_ranges: array containing parameter ranges
    This is a function only ever accessed by lnpostfn
    """
    for i in range(len(p)):
        if isinstance(par_ranges[i][0], str) or isinstance(par_ranges[i][1], str):
            logging.error(f"Bad param_ranges at index {i}: {par_ranges[i]}")
            raise ValueError("Non-numeric bounds detected in param_ranges")
        if p[i] < par_ranges[i][0] or p[i] > par_ranges[i][1]:
            return -np.inf
    return 0.0

def lnpostfn(params, param_ranges, fitType, start, step, numsteps, nxy, dxy, u, v, re, im, w):
    """
    Implement the posterior function. Some unit conversions are performed in order to keep galario happy 
    and also make the user-facing setup in nice thinkable units. 

    params: list containing the parameter vector
    param_ranges: array containing parameter ranges
    fitType: a string indicating which fit profile to use
    start, step, numsteps: origin, grid spacing, grid size of the radial grid to evaluate the function on (float)
    nxy: extent of fitted image in pixels. Powers of two preferred for speed (int)
    dxy: the pixel size of the image in radians (float)
    u, v: arrays of the u and v coordinates of the observed visibilities in units of wavelengths
    re, im, w: Re(V), Im(V), and weight of the observed visibilities
    """
    lnprior = lnpriorfn(params, param_ranges)  # apply prior
    if not np.isfinite(lnprior):
        return -np.inf
    if fitType=='gaussring':
        peak, sigma, ring_rad, inclination, posangle, dRA, dDec = params
        # convert from log to real space
        peak = 10**peak   
        # convert to radians
        arcsecs = [np.deg2rad(item/3600) for item in [sigma, ring_rad, start, step]]
        sigma, ring_rad, start, step = arcsecs
        degrees = [np.deg2rad(item) for item in [inclination, posangle, dRA, dDec]]
        inclination, posangle, dRA, dDec = degrees
        # compute the model brightness profile
        f = radial_gaussian_ring(peak, sigma, ring_rad, start, step, numsteps)
    elif fitType=='jinc':
        normalization, width, offset, inclination, posangle, dRA, dDec = params
        normalization = 10**normalization   # convert from log to real space
        offset = 10**offset
        # convert to radians
        arcsecs = [np.deg2rad(item/3600) for item in [width, start, step]]
        width, start, step = arcsecs
        degrees = [np.deg2rad(item) for item in [inclination, posangle, dRA, dDec]]
        inclination, posangle, dRA, dDec = degrees
        # compute the model brightness profile
        f = radial_jinc(normalization, width, offset, start, step, numsteps)
    else:
        logging.warning('Please choose a valid fitting model, or add a new one into the code.')
        return None

    chi2 = chi2Profile(f, start, step, nxy, dxy, u, v, re, im, w,
                       inc=inclination, PA=posangle, dRA=dRA, dDec=dDec)
    return -0.5 * chi2 #+ lnprior  #is zero if the code gets to this stage. If not using a uniform prior, take this into account here?
###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###

###%%%###%%%### Functions to do all the work ###%%%###%%%###
def read_param_file(file):
    """
    Read in parameters from the provided parameter file, with capabilities to read strings, floats, and ints. 
    file: string containing file path to a parameter file
    """
    params={}
    with open(file, 'r') as f:
        for line in f:
            line = line.split("#", 1)[0].strip() #This will skip in-line comments
            if "=" in line:
                pkey, pvalue = line.split("=", maxsplit=1)
                pkey = pkey.strip()
                pvalue = ast.literal_eval(pvalue.strip())
                if isinstance(pvalue, list):
                    pvalue = np.array(pvalue, dtype=float)
                params[pkey] = pvalue
    return params

def initialize_data(datatable, frequency):
    """
    Just a wrapper on galario get_image_size and the unpacking of my uvtable since I did those two together repeatedly. 
    datatable: string containing the filepath to a text file holding the observed visibility data in a uvtable (use uvplot package)
    frequency: observed frequency (float)
    """
    u, v, re, im, w = np.require(np.loadtxt(datatable, unpack=True), requirements='C')
    wavelength = 299792458/frequency
    u /= wavelength
    v /= wavelength

    nx, dx = get_image_size(u, v)
    return  nx, dx, u, v, re, im, w

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

    samples = sampler.chain[:, -1000:, :].reshape((-1, ndim))

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

def main(paramfile, fitType, ranges, guesses, productname):
    """
    Main function that takes care of the entire fitting process
    paramfile: string, path to the gloval parameter file
    fitType: string, indicating which fitting profile to use
    ranges: string, path to a csv file containing the allowed parameter ranges
    guesses: string, path to a csv file containing the initial parameter guesses
    productname: string, indicating what base filename to use when generating various products. 
    """
    #Set up the logger
    logging.basicConfig(filename=productname+'.log', filemode='w', level=logging.INFO)

    logging.info(paramfile)
    logging.info(fitType)
    logging.info(ranges)
    logging.info(guesses)
    logging.info(productname)

    logging.info('reading contents of range and guess files')
    #Need to read the contents of the range and guess files. 
    ranges = pd.read_csv(ranges, header=0, index_col=0).values
    guesses = pd.read_csv(guesses, header=0, index_col=0).values[:,0] #(if the shape is (n,1) that causes issues. Need (n,))

    logging.info('reading param file')
    paramdict = read_param_file(paramfile)
    data = initialize_data(paramdict['visFile'], paramdict['obsFreq'])
    ndim=len(ranges)
  
    logging.info('beginning pool')
    with Pool() as pool:
        sampler=es(paramdict['numWalkers'], ndim, lnpostfn, pool=pool, #note here that if you want a speed-up, eliminate the args passed in here (or at least make it such that the large arrays such as those in data are not passed. Memmap?)
                   args=[ranges, fitType, paramdict['radiusStart'], paramdict['radiusStep'], paramdict['radiusNumSteps'], data[0], data[1], data[2], data[3], data[4], data[5], data[6]]) #In testing, global variables has provided a notable increase in speed, but it is not a fantastic way of making this.
        pos = [guesses + 1e-5*np.random.randn(ndim) for i in range(paramdict['numWalkers'])]
        logging.info('Beginning the burn-in...')
        bi_start = time.time()
        pos0, _, _ = sampler.run_mcmc(pos, paramdict['burnTime'])
        sampler.reset()
        bi_end=time.time()
        logging.info("Duration of the burn-in is {0:.1f} seconds".format(bi_end-bi_start))
        logging.info('Beginning the fitting run...')
        fit_start=time.time()
        pos1, _, _ = sampler.run_mcmc(pos0, paramdict['runTime']) #change to pos1, prob1, state1 if you want to use those vars
        fit_end=time.time()
        logging.info("Duration of the fitting run is {0:.1f} seconds".format(fit_end-fit_start))

    print('saving results')
    save_results(fitType, sampler, pos1, productname)
 
    print('running visualization')
    #run the visualization
    visualize_main(paramdict, fitType, productname+'_pos.txt', productname)

###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###

###%%%###%%%### Call main, wrap in name=main to prevent recursion ###%%%###%%%###
if __name__=='__main__':
    print('running main')
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
