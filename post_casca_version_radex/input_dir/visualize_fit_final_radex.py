from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
from matplotlib import rc
from galario.double import sampleProfile, sampleImage, get_image_size, deg, arcsec
import os
from casatools import table
from casatasks import exportfits, imsubimage, tclean, split, impbcor, imsmooth

import argparse
import emcee
from model_vis_pfk_radex import model_visibility
from model_plots_pfk_radex import model_label, model_plot
from matplotlib.backends.backend_pdf import PdfPages
import logging
import corner

rc('text', usetex=True)
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : '14'}
rc('font', **font)

def jybm_to_jysr(infile):
    from astropy.io import fits

    hdr = fits.open(infile)[0].header
    bmaj = hdr['BMAJ']
    bmin = hdr['BMIN']

    omega_bm_deg2 = (np.pi/(2*np.log(2))) * bmaj*bmin
    omega_bm_sr = omega_bm_deg2 / (180/np.pi)**2
    return omega_bm_sr #divide the Jy/bm measurement by this factor

def img_prepper(fitsimg):
    #settings to control the section of the image that is plotted
    center = SkyCoord('10h43m57.75s', '11:42:13.34deg', frame='icrs')
    box_bkg = [40*u.arcsecond,40*u.arcsecond]
    box_contour1 = [40*u.arcsecond,40*u.arcsecond]

    im = fits.open(fitsimg)[0]
    im_wcs = WCS(im.header, naxis=2)
    conv = jybm_to_jysr(infile=fitsimg)
    im_plot = Cutout2D(im.data/conv, center, box_bkg, wcs=im_wcs)

    return im_wcs, im_plot.data

##################################################################################################################

def main():

    parser=argparse.ArgumentParser()
    parser.add_argument("fittype", choices=["gaussring", "twodgaussring", "twodgaussring_blob", "fixring_blob", "twodring_2blob_ne", "blob_radex15", "blob_radex6"], default="gaussring", type=str, help="Model as specified in model_profiles.py")
    pargs=parser.parse_args()

    fittype = pargs.fittype
    logging.info(fittype)

    logfile = './output/'+fittype+'_mcmc.log'
    logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)
    logging.info('Plotting results...')

    chain = './output/'+fittype+'_chain.hdf5'

    ms = './casa_dir/M95_C5+C2_cont93.ms'
    msx = './casa_dir/M95_C5+C2_cont93_trimmmedXX.ms'
    msy = './casa_dir/M95_C5+C2_cont93_trimmmedYY.ms'


    region_cut = './casa_dir/imsize_matcher.crtf'
    data_imgname = './casa_dir/M95_cont93GHz_auto'
    resid_imgname = './casa_dir/M95_cont93GHz_residual'

    pdf_plot = './output/'+fittype+'_plots.pdf'
    logging.info(f'Plot File: {pdf_plot}')
    pp = PdfPages(pdf_plot)

    reader = emcee.backends.HDFBackend(chain)
    samples = reader.get_chain()
    nsteps, nwalkers, ndim = samples.shape

    logging.info('Read in hdf5 chain file...')
    logging.info("Samples shape: %d steps, %d walkers, %d dims", nsteps, nwalkers, ndim)

    label = model_label(fittype)
    logging.info(label)
    if len(label) != ndim:
        logging.error(f"Dimension mismatch! File has {ndim} params, but labels has {len(label)}")
        return None

    flat_samples = reader.get_chain(discard=2500, flat=True)

    num_pars = flat_samples.shape[1]
    logging.info(f'{num_pars} pars')
    pars_bf = np.zeros(num_pars)
    for i in range(num_pars):
        pars_bf[i] = np.percentile(flat_samples[:,i], 50)
    logging.info(pars_bf)

    #paths plot
    fig, axes = plt.subplots(ndim, figsize=(24, 18), sharex=True)

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3, rasterized=True)
        ax.set_xlim(0, nsteps)
        ax.set_ylabel(label[i])

    axes[0].set_title(fittype+' paths')
    axes[-1].set_xlabel("step number")
    fig.tight_layout()
    pp.savefig()
    logging.info('Successfully saved the paths plot...')

    #corner plot
    try:
        fig = corner.corner(flat_samples, labels=label,
                    show_titles=True, quantiles=[0.16, 0.50, 0.84],
                    label_kwargs={'labelpad':20, 'fontsize':0}, fontsize=8)
        pp.savefig()
        logging.info('Successfully saved the corner plot...')
    except:
        #Most likely a TeX error, so we want the extra info
        logging.error('Something went wrong in the creation of the corner plot. Moving on...', exc_info=True)

##################################################################################################################
    #vis plot

    u_datx, v_datx, Re_datx, Im_datx, w_datx = np.require(np.loadtxt(msx+'.uvtable.txt', unpack=True), requirements='C')
    wavelength = 299792458/93e9
    u_datx /= wavelength
    v_datx /= wavelength

    vis_datx = np.array(Re_datx + 1j*Im_datx, dtype=np.complex256)

    u_daty, v_daty, Re_daty, Im_daty, w_daty = np.require(np.loadtxt(msy+'.uvtable.txt', unpack=True), requirements='C')
    u_daty /= wavelength
    v_daty /= wavelength

    vis_daty = np.array(Re_daty + 1j*Im_daty, dtype=np.complex256)

##################################################################################################################

    nxy, dxy = get_image_size(u_datx, v_datx, verbose=True)
    Rmin = 0
    dR = 0.0025
    nR = 10000 #stealing the values that I used for the radius vector when I performed the fit

    args_vis = (Rmin, dR, nR, nxy, dxy)
    vis_x_dat = (u_datx, v_datx, Re_datx, Im_datx, w_datx)

    logging.info(fittype)
    model = model_visibility(pars_bf, args_vis, vis_x_dat, fittype)
    logging.info('successfully loaded in model visibilities for plotting')

    resid_visx = vis_datx - model
    resid_visy = vis_daty - model

    logging.info('Residual visibilities calculated')

    tb = table()

    tb.open(msx)
    msdata = tb.getcol('DATA')
    tb.close()

    msdata_shape = msdata.shape
    implant_visx = np.reshape(resid_visx, msdata_shape)
    implant_visy = np.reshape(resid_visy, msdata_shape)

    #Do once for XX corr
    os.system('rm -rf '+msx+'.residual.ms')
    os.system('cp -R '+msx+' '+msx+'.residual.ms')
    tb.open(msx+'.residual.ms', nomodify=False)
    tb.putcol('DATA', implant_visx)
    tb.flush() #pushes changes to disk
    tb.close()

    #Repeat for YY corr. The fit was done to the XX data, but we want both correlations in order to increase SNR in the fitted image. 
    os.system('rm -rf '+msy+'.residual.ms')
    os.system('cp -R '+msy+' '+msy+'.residual.ms')
    tb.open(msy+'.residual.ms', nomodify=False)
    tb.putcol('DATA', implant_visy)
    tb.flush() #pushes changes to disk
    tb.close()

    logging.info('Residuals put into .residual.ms')

    os.system('rm -rf '+data_imgname+'.*')

    tclean(vis=ms,
        datacolumn='data',
        imagename=data_imgname,
        imsize=[1440, 1440],
        cell='0.075arcsec',
        specmode='mfs',
        gridder='standard',
        deconvolver='hogbom',
        restoringbeam='common',
        weighting='briggs',
        robust=0.5,
        niter=10000,
        interactive=False,
        threshold='5.52e-5 Jy',
        mask='circle[[720pix,722pix],125pix]')

    os.system('rm -rf '+resid_imgname+'.*')

    tclean(vis=[msx+'.residual.ms', msy+'.residual.ms'],
        datacolumn='data',
        imagename=resid_imgname,
        imsize=[1440, 1440],
        cell='0.075arcsec',
        specmode='mfs',
        gridder='standard',
        deconvolver='hogbom',
        restoringbeam='common',
        weighting='briggs',
        robust=0.5,
        niter=10000,
        interactive=False,
        threshold='5.73e-5 Jy',
        mask='circle[[720pix,722pix],125pix]')

    os.system('rm -rf '+data_imgname+'.image.pbcor')
    os.system('rm -rf '+data_imgname+'.image.pbcor.subim')
    os.system('rm -rf '+data_imgname+'.fits')
    os.system('rm -rf '+resid_imgname+'.image.pbcor')
    os.system('rm -rf '+resid_imgname+'.image.pbcor.subim')
    os.system('rm -rf '+resid_imgname+'.fits')

    for img in [data_imgname, resid_imgname]:
        impbcor(img+'.image', pbimage=img+'.pb', outfile=img+'.image.pbcor')
        imsubimage(img+'.image.pbcor', region=region_cut, outfile=img+'.image.pbcor.subim')
        exportfits(img+'.image.pbcor.subim', fitsimage=img+'.fits', dropdeg=True)

##################################################################################################################
    logging.info('tcleaning done')

    data_wcs, data_plot = img_prepper(data_imgname+'.fits')
    resid_wcs, resid_plot = img_prepper(resid_imgname+'.fits')

    #find fitted params for making the model
    #pos = np.loadtxt(fit_pos_array)
    #ndim=pos.shape[1]
    #medianfit = [np.percentile(pos[:, i], 50) for i in range(ndim)]
    #find the data image properties for making the model
    stealhdr = fits.open(data_imgname+'.fits')[0].header
    numpix = stealhdr['NAXIS2']
    pixarcsec = stealhdr['CDELT2']*3600

    args_plot = (numpix, pixarcsec)

    logging.info(fittype)
    ring_model = model_plot(pars_bf, args_plot, fittype)
    logging.info('successfully loaded in ring model for plotting')

    ring_model_plot = Cutout2D(ring_model, SkyCoord('10h43m57.75s', '11:42:13.34deg', frame='icrs'), [40*u.arcsecond,40*u.arcsecond], wcs=data_wcs).data

    #convert all from Jy/sr to MJy/sr
    data_plot /= 1e6
    resid_plot /= 1e6
    ring_model_plot /= 1e6

    fig = plt.figure(figsize=(24,8))

    #plot each, set plot labels and individual colorbars (necessary for plt3).
    ax1 = plt.subplot(131, projection=data_wcs)
    im1 = ax1.imshow(data_plot, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    ax1.text(5, 5, 'CLEAN Image', color='w', fontsize=16)
    cbar1 = plt.colorbar(mappable=im1, ax=ax1, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax2 = plt.subplot(132, projection=data_wcs)
    im2 = ax2.imshow(ring_model_plot, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    ax2.text(5, 5, 'Model Sky Intensity', color='w', fontsize=16)
    cbar2 = plt.colorbar(mappable=im2, ax=ax2, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax3 = plt.subplot(133, projection=data_wcs)
    im3 = ax3.imshow(resid_plot, vmin=np.percentile(resid_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    ax3.text(5, 5, 'CLEAN Residual Visibilities', color='black', fontsize=16)
    cbar3 = plt.colorbar(mappable=im3, ax=ax3, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    #Mess with axis labels
    axes=[ax1, ax2, ax3]
    for ax in axes:
        if ax != ax1:
            ax.set_ylabel(r'', size=0)
            ax.coords[1].set_ticklabel(size=0)
        else:
            ax.set_ylabel(r"Declination (J2000)", size=22,labelpad=1)
        if ax != ax2:
            ax.set_xlabel(r'', size=0)
        else:
            ax.set_xlabel(r"Right Ascension (J2000)", size=22)

    #I wanted to play around with adding contours, but the difference in image noise is proving problematic. Need to fix that before these will be helpful. 
    #ax1.contour(data_plot, data_plot, colors='w', transform=ax1.get_transform(data_wcs), levels=2.15e-5/jybm_to_jysr('modelling/cont93GHz/M95_cont93GHz.fits')/1e6*np.array([2.5, 4, 6]), zorder=10)
    #ax3.contour(resid_plot, resid_plot, colors='yellow', transform=ax3.get_transform(resid_wcs), levels=2.1e-5/jybm_to_jysr('modelling/cont93GHz/M95_cont93GHz_residual.fits')/1e6*np.array([3, 9]), zorder=5)

    fig.suptitle("93GHZ Continuum Intensity of NGC 3351 (MJy/sr)", size=30)
    fig.subplots_adjust(hspace=0.1,wspace=0.1)

    pp.savefig()

##################################################################################################################
    #vis plot contoured

    fig = plt.figure(figsize=(24,8))

    #plot each, set plot labels and individual colorbars (necessary for plt3).
    ax1 = plt.subplot(131, projection=data_wcs)
    im1 = ax1.imshow(data_plot, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    ax1.text(5, 5, 'CLEAN Image', color='w', fontsize=16)
    cbar1 = plt.colorbar(mappable=im1, ax=ax1, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax2 = plt.subplot(132, projection=data_wcs)
    im2 = ax2.imshow(ring_model_plot, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    ax2.text(5, 5, 'Model Sky Intensity', color='w', fontsize=16)
    cbar2 = plt.colorbar(mappable=im2, ax=ax2, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax3 = plt.subplot(133, projection=data_wcs)
    im3 = ax3.imshow(resid_plot, vmin=np.percentile(resid_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    ax3.text(5, 5, 'CLEAN Residual Visibilities', color='black', fontsize=16)
    cbar3 = plt.colorbar(mappable=im3, ax=ax3, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    peak_flux = np.percentile(data_plot, 99.95)
    #print(peak_flux)

    ax1.contour(ring_model_plot, ring_model_plot, colors='w', transform=ax1.get_transform(data_wcs), levels=peak_flux*np.array([0.2, 0.4, 0.6, 0.8, 0.95]), zorder=10, linewidths=0.5)
    ax2.contour(ring_model_plot, ring_model_plot, colors='k', transform=ax2.get_transform(data_wcs), levels=peak_flux*np.array([0.2, 0.4, 0.6, 0.8, 0.95]), zorder=10, linewidths=0.5)
    ax3.contour(ring_model_plot, ring_model_plot, colors='k', transform=ax3.get_transform(data_wcs), levels=peak_flux*np.array([0.2, 0.4, 0.6, 0.8, 0.95]), zorder=10, linewidths=0.5)

    #Mess with axis labels
    axes=[ax1, ax2, ax3]
    for ax in axes:
        if ax != ax1:
            ax.set_ylabel(r'', size=0)
            ax.coords[1].set_ticklabel(size=0)
        else:
            ax.set_ylabel(r"Declination (J2000)", size=22,labelpad=1)
        if ax != ax2:
            ax.set_xlabel(r'', size=0)
        else:
            ax.set_xlabel(r"Right Ascension (J2000)", size=22)

    #I wanted to play around with adding contours, but the difference in image noise is proving problematic. Need to fix that before these will be helpful. 
    #ax1.contour(data_plot, data_plot, colors='w', transform=ax1.get_transform(data_wcs), levels=2.15e-5/jybm_to_jysr('modelling/cont93GHz/M95_cont93GHz.fits')/1e6*np.array([2.5, 4, 6]), zorder=10)
    #ax3.contour(resid_plot, resid_plot, colors='yellow', transform=ax3.get_transform(resid_wcs), levels=2.1e-5/jybm_to_jysr('modelling/cont93GHz/M95_cont93GHz_residual.fits')/1e6*np.array([3, 9]), zorder=5)

    fig.suptitle("93GHZ Continuum Intensity of NGC 3351 (MJy/sr)", size=30)
    fig.subplots_adjust(hspace=0.1,wspace=0.1)

    pp.savefig()

    pp.close()

    logging.info('Done!')

if __name__=='__main__':
    main()