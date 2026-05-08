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
from model_profiles_pfk import gaussring_model, gaussring_1blob_model

rc('text', usetex=True)
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : '14'}
rc('font', **font)

def radial_gaussian_ring(peak_ring, sigma_ring, rad_ring, start, step, numsteps):
    radius = np.linspace(start, start + numsteps*step, numsteps)
    return 10**peak_ring * np.exp((-1/2)*((radius-rad_ring)/sigma_ring)**2)

def radial_gaussian_ring_plot(peak_ring, sigma_ring, rad_ring, radius):
    return 10**peak_ring*np.exp((-1/2)*((radius-rad_ring)/sigma_ring)**2)

def gauss_blob_plot(peak_blob, sigma_blob, xx, yy, xoff, yoff):
    return 10**peak_blob * np.exp((-1/2) * (((xx-xoff)/sigma_blob)**2 + ((yy-yoff)/sigma_blob)**2))

def make_model_ring(npix, pixscale, peak, sigma, ringrad, inc, pa, dra, ddec):
    import numpy as np
    from astropy import units as u

    #The inputs that are the product of the emcee fitting will come out in the units in which I put them into the fitting chain.
    #This means that the peak is in Jy/sr, the sigma and ringrad are in arcseconds, and inc, pa, deltaRA and deltaDec are in degrees. 

    image_size = npix * pixscale
    x = np.linspace(-image_size/2, image_size/2, npix)
    y = np.linspace(-image_size/2, image_size/2, npix)
    xx, yy = np.meshgrid(x, y)

    #apply deltaRA and deltaDec
    xx_shifted = xx + (dra) #convert to arcsec, then pix (to match xx system) #move left 
    yy_shifted = yy - (ddec) #move up

    xpa = xx_shifted*np.cos(np.deg2rad(pa)) + yy_shifted*np.sin(np.deg2rad(pa))
    ypa = -xx_shifted*np.sin(np.deg2rad(pa)) + yy_shifted*np.cos(np.deg2rad(pa))
    
    xinc = xpa/np.cos(np.deg2rad(inc))

    rad = np.hypot(xinc, ypa)
    model_jysr = radial_gaussian_ring_plot(peak, sigma, ringrad, rad)
    return model_jysr #leave in Jy/sr here. I will convert Jy/bm to Jy/sr to match. 


def make_model_blob(npix, pixscale, peak, sigma, ringrad, inc, pa, dra, ddec, peak_b, sigma_b, dist, ang):
    import numpy as np
    from astropy import units as u

    #The inputs that are the product of the emcee fitting will come out in the units in which I put them into the fitting chain.
    #This means that the peak is in Jy/sr, the sigma and ringrad are in arcseconds, and inc, pa, deltaRA and deltaDec are in degrees. 

    image_size = npix * pixscale
    x = np.linspace(-image_size/2, image_size/2, npix)
    y = np.linspace(-image_size/2, image_size/2, npix)
    xx, yy = np.meshgrid(x, y)

    #apply deltaRA and deltaDec
    xx_shifted = xx + (dra) #convert to arcsec, then pix (to match xx system) #move left 
    yy_shifted = yy - (ddec) #move up

    xpa = xx_shifted*np.cos(np.deg2rad(pa)) + yy_shifted*np.sin(np.deg2rad(pa))
    ypa = -xx_shifted*np.sin(np.deg2rad(pa)) + yy_shifted*np.cos(np.deg2rad(pa))
    
    #ring
    xinc = xpa/np.cos(np.deg2rad(inc))
    rad = np.hypot(xinc, ypa)

    #blob
    xdot = dist * np.sin(np.deg2rad(ang)) * np.cos(np.deg2rad(inc))
    ydot = dist * np.cos(np.deg2rad(ang))

    ring_model = radial_gaussian_ring_plot(peak, sigma, ringrad, rad)
    blob_model = gauss_blob_plot(peak_b, sigma_b, xpa, ypa, xdot, ydot)
    model_jysr = ring_model+blob_model
    
    return model_jysr #leave in Jy/sr here. I will convert Jy/bm to Jy/sr to match. 

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

###

def main():

    print('begin!')

    parser=argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="Path to product directory")
    parser.add_argument("fittype", choices=["gaussring", "twodgaussring", "twodgaussring_blob", "fixring_blob"], default="gaussring", type=str, help="Model as specified in model_profiles.py")
    parser.add_argument("casapath", type=str, help="Path to image directory")
    pargs=parser.parse_args()

    filepath = pargs.filepath
    fittype = pargs.fittype
    casapath = pargs.casapath

    ms = casapath+'/M95_C5+C2_cont93.ms'
    msx = casapath+'/M95_C5+C2_cont93_trimmmedXX.ms'
    msy = casapath+'/M95_C5+C2_cont93_trimmmedYY.ms'

    region_cut = casapath+'/imsize_matcher.crtf'
    data_imgname = casapath+'/M95_cont93GHz_auto'
    resid_imgname = casapath+'/M95_cont93GHz_residual'

    ###

    print('1!')

    u_datx, v_datx, Re_datx, Im_datx, w_datx = np.require(np.loadtxt(msx+'.uvtable.txt', unpack=True), requirements='C')
    wavelength = 299792458/93e9
    u_datx /= wavelength
    v_datx /= wavelength

    vis_datx = np.array(Re_datx + 1j*Im_datx, dtype=np.complex256)

    u_daty, v_daty, Re_daty, Im_daty, w_daty = np.require(np.loadtxt(msy+'.uvtable.txt', unpack=True), requirements='C')
    u_daty /= wavelength
    v_daty /= wavelength

    vis_daty = np.array(Re_daty + 1j*Im_daty, dtype=np.complex256)

    ###

    print('2!')

    nxy, dxy = get_image_size(u_datx, v_datx, verbose=True)
    Rmin = 0
    dR = 0.0025
    nR = 10000 #stealing the values that I used for the radius vector when I performed the fit

    ###
    print('3!')

    #fit_pos_array = '/arc/home/pknowlton/uv_product_dir/test0/gaussring_row0_chain.txt'

    #pos = np.loadtxt(fit_pos_array)
    #ndim=pos.shape[1]
    #med = [np.percentile(pos[:, i], 50) for i in range(ndim)]

    #get chain info

    chain = filepath+'_chain.hdf5'

    reader = emcee.backends.HDFBackend(chain)
    samples = reader.get_chain()
    flat_samples = reader.get_chain(discard=2500, flat=True)

    num_pars = flat_samples.shape[1]
    print(num_pars, ' pars')
    pars_bf = np.zeros(num_pars)

    for i in range(num_pars):
        pars_bf[i] = np.percentile(flat_samples[:,i], 50)

    print(pars_bf)

    ###

    #sigma_rad = pars_bf[1] * arcsec
    #ring_rad = pars_bf[2] * arcsec
    
    #inc_rad = pars_bf[3] * deg
    #pa_rad = pars_bf[4]* deg

    #dra_rad = pars_bf[5] * arcsec
    #ddec_rad = pars_bf[6] * arcsec

    if fittype == 'gaussring':

        print('gaussring')
        sample = radial_gaussian_ring(pars_bf[0], pars_bf[1], pars_bf[2], Rmin, dR, nR)

        Rmin /= 206265 #converting to rad for use in sampleProfile
        dR /=206265
        
        model = np.array(sampleProfile(sample, Rmin, dR, nxy, dxy, u_datx, v_datx, inc=inc_rad, PA=pa_rad, dRA=dra_rad, dDec=ddec_rad), dtype=np.complex256)

    elif fittype == 'twodgaussring':

        print('twodgaussring')

        sigma_rad = pars_bf[1] * arcsec
        ring_rad = pars_bf[2] * arcsec

        inc_rad = pars_bf[3] * deg
        pa_rad = pars_bf[4]* deg

        dra_rad = pars_bf[5] * arcsec
        ddec_rad = pars_bf[6] * arcsec

        model_img = gaussring_model(pars_bf[0], sigma_rad, ring_rad, inc_rad, nxy, dxy)
        model = np.array(sampleImage(model_img, dxy, u_datx, v_datx, dRA=dra_rad, dDec=ddec_rad, PA=pa_rad, origin='lower'), dtype=np.complex256)

    elif fittype == 'twodgaussring_blob':

        print('twodgaussring_blob')
        
        sigma_rad = pars_bf[1] * arcsec
        ring_rad = pars_bf[2] * arcsec

        inc_rad = pars_bf[3] * deg
        pa_rad = pars_bf[4]* deg

        dra_rad = pars_bf[5] * arcsec
        ddec_rad = pars_bf[6] * arcsec

        sigma_b_rad = pars_bf[8] * arcsec
        dist_rad = pars_bf[9] * arcsec

        ang_rad = pars_bf[10] * deg

        model_img = gaussring_1blob_model(pars_bf[0], sigma_rad, ring_rad, inc_rad, pars_bf[7], sigma_b_rad, dist_rad, ang_rad, nxy, dxy)
        model = np.array(sampleImage(model_img, dxy, u_datx, v_datx, dRA=dra_rad, dDec=ddec_rad, PA=pa_rad, origin='lower'), dtype=np.complex256)

    elif fittype == 'fixring_blob':

        print('fixring_blob')
        
        sigma_rad = 0.94 * arcsec
        ring_rad = 6.97 * arcsec

        inc_rad = 61.98 * deg
        pa_rad = 13.84 * deg

        dra_rad = 0.29 * arcsec
        ddec_rad = 0.38 * arcsec

        sigma_b_rad = pars_bf[1] * arcsec
        dist_rad = pars_bf[2] * arcsec

        ang_rad = pars_bf[3] * deg

        model_img = gaussring_1blob_model(6.54, sigma_rad, ring_rad, inc_rad, pars_bf[0], sigma_b_rad, dist_rad, ang_rad, nxy, dxy)
        model = np.array(sampleImage(model_img, dxy, u_datx, v_datx, dRA=dra_rad, dDec=ddec_rad, PA=pa_rad, origin='lower'), dtype=np.complex256)
    
    else:
        print('Please choose a valid fitting model, or add a new one into the code.')

    ###
    print('4!')

    resid_visx = vis_datx - model
    resid_visy = vis_daty - model

    ###
    print('5!')

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

    ###
    print('6!')

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

    ###
    print('7!')

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

    if fittype == 'gaussring':

        print('gaussring')
        ring_model = make_model_ring(numpix, pixarcsec, pars_bf[0], pars_bf[1], pars_bf[2], pars_bf[3], pars_bf[4], pars_bf[5], pars_bf[6])

    elif fittype == 'twodgaussring':

        print('twodgaussring')
        ring_model = make_model_ring(numpix, pixarcsec, pars_bf[0], pars_bf[1], pars_bf[2], pars_bf[3], pars_bf[4], pars_bf[5], pars_bf[6])

    elif fittype == 'twodgaussring_blob':

        print('twodgaussring_blob')
        ring_model = make_model_blob(numpix, pixarcsec, pars_bf[0], pars_bf[1], pars_bf[2], pars_bf[3], pars_bf[4], pars_bf[5], pars_bf[6], pars_bf[7], pars_bf[8], pars_bf[9], pars_bf[10])

    elif fittype == 'fixring_blob':

        print('fixring_blob')
        ring_model = make_model_blob(numpix, pixarcsec, 6.54, 0.94, 6.97, 61.98, 13.84, 0.29, 0.38, pars_bf[0], pars_bf[1], pars_bf[2], pars_bf[3])
    
    else:
        print('Please choose a valid fitting model, or add a new one into the code.')

    #actually generate the 2d model array
    #ring_model = make_model_ring(numpix, pixarcsec, pars_bf[0], pars_bf[1], pars_bf[2], pars_bf[3], pars_bf[4], pars_bf[5], pars_bf[6])
    #ring_model = make_model_ring(numpix, pixarcsec, 6.54, 0.94, 6.97, 61.98, 13.84, 0.29, 0.38)
    #make sure it is same dimensions, position as the other two. 
    ring_model_plot = Cutout2D(ring_model, SkyCoord('10h43m57.75s', '11:42:13.34deg', frame='icrs'), [40*u.arcsecond,40*u.arcsecond], wcs=data_wcs).data

    #convert all from Jy/sr to MJy/sr
    data_plot /= 1e6
    resid_plot /= 1e6
    ring_model_plot /= 1e6

    fig = plt.figure(figsize=(16,5))

    #plot each, set plot labels and individual colorbars (necessary for plt3).
    ax1 = plt.subplot(131, projection=data_wcs)
    im1 = ax1.imshow(data_plot, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno')
    ax1.text(5, 5, 'CLEAN Image', color='w', fontsize=16)
    cbar1 = plt.colorbar(mappable=im1, ax=ax1, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax2 = plt.subplot(132, projection=data_wcs)
    im2 = ax2.imshow(ring_model_plot, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno')
    ax2.text(5, 5, 'Model Sky Intensity', color='w', fontsize=16)
    cbar2 = plt.colorbar(mappable=im2, ax=ax2, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax3 = plt.subplot(133, projection=data_wcs)
    im3 = ax3.imshow(resid_plot, vmin=np.percentile(resid_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='coolwarm')
    ax3.text(5, 5, 'CLEAN Residual Visibilities', color='black', fontsize=16)
    cbar3 = plt.colorbar(mappable=im3, ax=ax3, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    peak_flux = np.nanmax(ring_model_plot)
    print(peak_flux)
    peak_flux_dat = np.nanmax(data_plot)
    print(peak_flux_dat)

    ax1.contour(ring_model_plot, ring_model_plot, colors='w', transform=ax1.get_transform(data_wcs), levels=peak_flux*np.array([0.05, 0.1, 0.2, 0.5, 0.8]), zorder=10, linewidths=0.5)
    ax2.contour(ring_model_plot, ring_model_plot, colors='k', transform=ax2.get_transform(data_wcs), levels=peak_flux*np.array([0.05, 0.1, 0.2, 0.5, 0.8]), zorder=10, linewidths=0.5)
    ax3.contour(ring_model_plot, ring_model_plot, colors='k', transform=ax3.get_transform(data_wcs), levels=peak_flux*np.array([0.05, 0.1, 0.2, 0.5, 0.8]), zorder=10, linewidths=0.5)

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

    comp_plot = filepath+'_compplot_contour.pdf'
    plt.savefig(fname=comp_plot, bbox_inches='tight')

    print('done!')

if __name__=='__main__':
    main()