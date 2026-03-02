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

def save_results_hdf(fittype, outname, chainname):

    reader = emcee.backends.HDFBackend(chainname)
    samples = reader.get_chain()
    nsteps, nwalkers, ndim = samples.shape
    print('saved chain')
    print(nsteps, nwalkers, ndim)

    if fittype=='gaussring':
        #label = ["Peak", "$\sigma$", r"R$_{ring}$", "Inc", "PA", r"$\Delta$RA", r"$\Delta$Dec"]
        label = ["Peak", "Width", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec"]
        print('gaussring')
    elif fittype=='jinc':
        #label = ["Peak", "Width", "Offset", "Inc", "PA", r"$\Delta$RA", r"$\Delta$Dec"]
        label = ["Peak", "Width", "Offset", "Inc", "PA", "Offset RA", "Offset Dec"]
    else:
        #logging.warning('Please choose a valid fitting model, or add a new one into the code.')
        print('invalid')
        return None

    if len(label) != ndim:
        #logging.error(f"Dimension mismatch! File has {ndim} params, but labels has {len(label)}")
        print('wrong dims')
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
    #logging.info('Successfully saved the paths plot')
    print('saved paths plot')

    #corner plot
    flat_samples = reader.get_chain(discard=150, flat=True)
    try:
        fig = corner.corner(flat_samples, labels=label,
                    show_titles=True, quantiles=[0.16, 0.50, 0.84],
                    label_kwargs={'labelpad':20, 'fontsize':0}, fontsize=8)
        fig.savefig(outname+'_corner.png')
        #logging.info('Successfully saved the corner plot')
        print('saved corner plot')
    except:
        #Most likely a TeX error, so we want the extra info
        print('didnt save corner plot')

def main():

    #well get rid of this soon
    print('running')
    productname = '/arc/home/pknowlton/uv_product_dir/test4/gaussing_test4_full'
    chain = productname+'_chain.hdf5'
    fittype='gaussring'

    save_results_hdf(fittype, productname, chain)
    print('done!')

if __name__=='__main__':
    main()