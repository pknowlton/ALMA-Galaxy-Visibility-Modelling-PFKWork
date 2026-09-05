"""
Dynesty Visualization, Residual Imaging, and Post-Processing
=============================================================
This script post-processes the results of a Dynesty nested sampling run:
1. Restores the sampling state from the checkpoint file (`<fittype>_checkpoint.save`).
2. Extracts parameter posteriors and computes best-fit (median) parameter estimates.
3. Generates Dynesty diagnostic figures (run progression summary and corner plot).
4. Evaluates best-fit synthetic visibilities and subtracts them from the observed data
   to produce residual visibilities: $V_{\text{resid}} = V_{\text{obs}} - V_{\text{mod}}$.
5. Implants residual visibilities into CASA Measurement Sets (`.ms`) and performs
   synthesis deconvolution (`tclean`) and primary beam correction (`impbcor`).
6. Generates multi-panel comparison figures showing:
   [Observed CLEAN Image | Model Sky Intensity | CLEAN Residual Visibilities]
   with and without contour overlays, saving all figures to a multipage PDF.

Requirements:
    - Python environment with Dynesty, CASA (casatools, casatasks), Astropy, and Matplotlib.
"""

import os
os.environ["CASACORE_MEASURES_AUTO_UPDATE"] = "false"

from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
from matplotlib import rc
from galario.double import get_image_size
from casatools import table
from casatasks import exportfits, imsubimage, tclean, split, impbcor, imsmooth

import argparse
import dynesty
from dynesty import DynamicNestedSampler
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
from model_prof import model_prof, model_addon
from radec_calc import sun_radec, dynest_radec, make_model_wcs
from matplotlib.backends.backend_pdf import PdfPages
import logging
import corner

# Enable LaTeX typesetting for figure labels and text annotations
rc('text', usetex=True)
font = {'family': 'serif',
        'weight': 'bold',
        'size': '14'}
rc('font', **font)

def jybm_to_jysr(infile):
    """
    Computes the beam solid angle factor to convert Jy/beam into Jy/steradian.

    Radio interferometric images produced by CLEAN deconvolution are calibrated in
    flux density per synthesised beam ($\text{Jy}/\text{beam}$). To obtain physical
    surface brightness ($\text{Jy}/\text{sr}$), the pixel values are divided by the
    beam solid angle:
        $\Omega_{\text{beam}} = \frac{\pi}{4 \ln 2} \cdot \theta_{\text{maj}} \cdot \theta_{\text{min}}$
    where $\theta_{\text{maj}}$ (`BMAJ`) and $\theta_{\text{min}}$ (`BMIN`) are the
    restoring beam major and minor full-width at half-maximum (FWHM) axes.

    Parameters
    ----------
    infile : str
        Path to the FITS image file containing `BMAJ` and `BMIN` in its primary header.

    Returns
    -------
    omega_bm_sr : float
        Synthesised beam solid angle in steradians.
    """
    from astropy.io import fits

    hdr = fits.open(infile)[0].header
    bmaj = hdr['BMAJ']  # Major axis FWHM in degrees
    bmin = hdr['BMIN']  # Minor axis FWHM in degrees

    # Gaussian beam solid angle in square degrees: pi / (4 * ln(2)) * FWHM_maj * FWHM_min
    omega_bm_deg2 = (np.pi / (2 * np.log(2))) * bmaj * bmin
    # Convert square degrees to steradians: 1 sr = (180 / pi)^2 deg^2
    omega_bm_sr = omega_bm_deg2 / (180 / np.pi)**2
    return omega_bm_sr  # Divide Jy/beam measurements by this factor to obtain Jy/sr

def img_prepper(fitsimg):
    """
    Loads a FITS image, extracts WCS coordinates, converts units, and crops a 2D cutout.

    Extracts a $40'' \times 40''$ field of view centered on the galactic nucleus of
    NGC 3351 ($\alpha = 10^{\text{h}}43^{\text{m}}57.75^{\text{s}}$, $\delta = +11^\circ 42' 13.34''$).

    Parameters
    ----------
    fitsimg : str
        Path to the input FITS image file.

    Returns
    -------
    im_wcs : astropy.wcs.WCS
        2D celestial World Coordinate System extracted from the FITS header.
    data : numpy.ndarray
        2D image cutout array in surface brightness units ($\text{Jy}/\text{sr}$).
    """
    # Coordinates of NGC 3351 galaxy nucleus and field-of-view cutout size
    center = SkyCoord('10h43m57.7330s', '+11d42m12.9996s', frame='icrs')
    box_bkg = [36 * u.arcsecond, 36 * u.arcsecond]

    im = fits.open(fitsimg)[0]
    im_wcs = WCS(im.header, naxis=2)
    logging.info(im_wcs)
    conv = jybm_to_jysr(infile=fitsimg)
    # Convert image from Jy/beam to Jy/sr and crop around galaxy center
    im_plot = Cutout2D(im.data / conv, center, box_bkg, wcs=im_wcs)

    return im_plot.wcs, im_plot.data

##################################################################################################################
##################################################################################################################
##################################################################################################################


def main():
    """
    Main visualization and post-processing routine.

    Workflow:
    1. Restores the Dynesty sampler from checkpoint file (`./output/<fittype>_checkpoint.save`).
    2. Calculates median parameter values (best-fit model $\hat{\theta}$).
    3. Plots nested sampling summary trace (`runplot`) and corner covariance posterior plot (`cornerplot`).
    4. Computes model visibilities for XX and YY polarization baselines using Galario.
    5. Calculates residual visibilities ($V_{\text{obs}} - V_{\text{mod}}$) and writes them
       into temporary CASA Measurement Sets (`.residual.ms`).
    6. Runs CASA `tclean` imaging and primary beam correction (`impbcor`) on both data and residuals.
    7. Evaluates the 2D model sky brightness map and plots a 3-panel comparison figure
       (Observed Data, Best-Fit Model, Residual Map) in $\text{MJy}/\text{sr}$.
    8. Generates a second 3-panel figure featuring model surface brightness contour overlays.
    9. Compiles all figures into a multipage PDF (`./output/<fittype>_plots.pdf`).
    """
    parser = argparse.ArgumentParser(
        description="Visualize Dynesty results, compute visibility residuals, and generate CASA CLEAN images."
    )
    parser.add_argument("fittype", type=str, help="Model configuration identifier (e.g. 'twod_gaussring').")
    pargs = parser.parse_args()

    fittype = pargs.fittype

    logfile = './output/' + fittype + '_dynesty.log'
    logging.basicConfig(
        filename=logfile,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )
    logging.info("==========Visualization Output==========")
    logging.info('Fittype: %s', fittype)

    # --------------------------------------------------------------------------
    # 1. Restore Dynesty Checkpoint & Parameter Posterior Analysis
    # --------------------------------------------------------------------------
    check_file = './output/' + fittype + '_checkpoint.save'
    sampler = DynamicNestedSampler.restore(check_file)
    res = sampler.results

    logging.info(f"Read in the results from the saved checkpoint file {check_file} ")

    labels, units, ndim = model_addon(fittype)

    logging.info(f"Dimensions for {fittype}: {ndim}")
    logging.info(f"Parameters for {fittype}: {labels}")

    pars_bf = np.zeros(ndim)
    logz = res.logz[-1]
    weights = np.exp(res.logwt - logz)
    
    # Calculate 16th, 50th (median), and 84th percentiles; assign median to pars_bf
    for i in range(ndim):
        quantiles = dyfunc.quantile(res.samples[:, i], [0.159, 0.5, 0.841], weights=weights)
        pars_bf[i] = quantiles[1]
    logging.info(f"Best fit parameters: {pars_bf}")

    pdf_plot = './output/' + fittype + '_plots.pdf'
    pp = PdfPages(pdf_plot)
    logging.info(f'Plot File: {pdf_plot}')

    # --------------------------------------------------------------------------
    # 2. Dynesty Run Summary & Corner Plots
    # --------------------------------------------------------------------------
    # Summary plot: Live points, log-evidence evolution, and posterior weights
    fig, axes = dyplot.runplot(res)
    fig.tight_layout()
    pp.savefig(fig)
    plt.close(fig)
    logging.info('Successfully saved the summary plot...')

    logging.info(f"Total samples: {len(weights)}")
    logging.info(f"Non-zero weight samples: {np.count_nonzero(weights)}")
    logging.info(f"Max weight / Sum weight: {np.max(weights) / np.sum(weights):.4f}")
        
    # Corner plot: 1D marginal posteriors and 2D joint covariance contours
    fig, axes = dyplot.cornerplot(
        res, labels=labels, quantiles=[0.159, 0.5, 0.841],
        color='dodgerblue', show_titles=True
    )
    pp.savefig(fig, dpi=300)
    plt.close(fig)
    logging.info('Successfully saved the corner plot...')

    # --------------------------------------------------------------------------
    # 3. Load Observed Visibilities for Residual Calculation
    # --------------------------------------------------------------------------
    ms = './casa_dir/M95_C5+C2_cont93.ms'
    msx = './casa_dir/M95_C5+C2_cont93_trimmmedXX.ms'
    msy = './casa_dir/M95_C5+C2_cont93_trimmmedYY.ms'

    data_imgname = './casa_dir/M95_cont93GHz_auto'
    resid_imgname = './casa_dir/M95_cont93GHz_residual'

    # Load XX baseline visibilities
    u_datx, v_datx, Re_datx, Im_datx, w_datx = np.require(np.loadtxt(msx + '.uvtable.txt', unpack=True), requirements='C')
    wavelength = 299792458 / 93e9
    u_datx /= wavelength
    v_datx /= wavelength
    vis_datx = np.array(Re_datx + 1j * Im_datx, dtype=np.complex256)

    # Load YY baseline visibilities
    u_daty, v_daty, Re_daty, Im_daty, w_daty = np.require(np.loadtxt(msy + '.uvtable.txt', unpack=True), requirements='C')
    u_daty /= wavelength
    v_daty /= wavelength
    vis_daty = np.array(Re_daty + 1j * Im_daty, dtype=np.complex256)

    # Compute Fourier grid dimensions
    nxy, dxy = get_image_size(u_datx, v_datx, verbose=True)
    dxy_arcsec = dxy * 206265
    args_vis = (nxy, dxy)
    vis_x_dat = (u_datx, v_datx, Re_datx, Im_datx, w_datx)

    # Sample best-fit model visibilities using Galario
    logging.info('Fittype: %s', fittype)
    model = model_prof(pars_bf, args_vis, vis_x_dat, 'vis', fittype)
    logging.info('Successfully computed best fit model visibilities...')

    # Compute residual complex visibilities
    resid_visx = vis_datx - model
    resid_visy = vis_daty - model
    logging.info('Residual visibilities calculated...')

    # --------------------------------------------------------------------------
    # 4. Implant Residual Visibilities into CASA Measurement Sets
    # --------------------------------------------------------------------------
    tb = table()

    tb.open(msx)
    msdata = tb.getcol('DATA')
    tb.close()

    msdata_shape = msdata.shape
    implant_visx = np.reshape(resid_visx, msdata_shape)
    implant_visy = np.reshape(resid_visy, msdata_shape)

    # Write residuals for XX polarization
    os.system('rm -rf ' + msx + '.residual.ms')
    os.system('cp -R ' + msx + ' ' + msx + '.residual.ms')
    tb.open(msx + '.residual.ms', nomodify=False)
    tb.putcol('DATA', implant_visx)
    tb.flush()  # Commit changes to disk
    tb.close()

    # Write residuals for YY polarization (increases SNR in synthesized image)
    os.system('rm -rf ' + msy + '.residual.ms')
    os.system('cp -R ' + msy + ' ' + msy + '.residual.ms')
    tb.open(msy + '.residual.ms', nomodify=False)
    tb.putcol('DATA', implant_visy)
    tb.flush()  # Commit changes to disk
    tb.close()

    logging.info('Residual data put into .residual.ms')

    # --------------------------------------------------------------------------
    # 5. CASA tclean Deconvolution and Primary Beam Correction
    # --------------------------------------------------------------------------
    # Clean observed data
    os.system('rm -rf ' + data_imgname + '.*')
    tclean(
        vis=ms,
        datacolumn='data',
        imagename=data_imgname,
        imsize=[nxy, nxy],
        cell=dxy_arcsec,
        specmode='mfs',
        gridder='standard',
        deconvolver='hogbom',
        restoringbeam='common',
        weighting='briggs',
        robust=0.5,
        niter=10000,
        interactive=False,
        threshold='5.52e-5 Jy',
        mask='circle[[2048pix,2052pix],400pix]'
    )

    # Clean residual visibilities
    os.system('rm -rf ' + resid_imgname + '.*')
    tclean(
        vis=[msx + '.residual.ms', msy + '.residual.ms'],
        datacolumn='data',
        imagename=resid_imgname,
        imsize=[nxy, nxy],
        cell=dxy_arcsec,
        specmode='mfs',
        gridder='standard',
        deconvolver='hogbom',
        restoringbeam='common',
        weighting='briggs',
        robust=0.5,
        niter=10000,
        interactive=False,
        threshold='5.73e-5 Jy',
        mask='circle[[2048pix,2052pix],400pix]'
    )

    # Primary beam correction and FITS image export
    os.system('rm -rf ' + data_imgname + '.image.pbcor')
    os.system('rm -rf ' + data_imgname + '.image.pbcor.subim')
    os.system('rm -rf ' + data_imgname + '.fits')
    os.system('rm -rf ' + resid_imgname + '.image.pbcor')
    os.system('rm -rf ' + resid_imgname + '.image.pbcor.subim')
    os.system('rm -rf ' + resid_imgname + '.fits')

    for img in [data_imgname, resid_imgname]:
        impbcor(img + '.image', pbimage=img + '.pb', outfile=img + '.image.pbcor')
        #imsubimage(img + '.image.pbcor', region=region_cut, outfile=img + '.image.pbcor.subim')
        exportfits(img + '.image.pbcor', fitsimage=img + '.fits', dropdeg=True)

    logging.info('CASA tcleaning data and residual data done...')

    # --------------------------------------------------------------------------
    # 6. Generate 2D Model Sky Map & Multi-Panel Comparison Figures
    # --------------------------------------------------------------------------
    data_wcs, data_plot, = img_prepper(data_imgname + '.fits')
    resid_wcs, resid_plot, = img_prepper(resid_imgname + '.fits')
    logging.info(data_wcs)
    logging.info(resid_wcs)

    # Extract pixel grid properties from the generated data FITS image header
    #stealhdr = fits.open(data_imgname + '.fits')[0].header
    #numpix = stealhdr['NAXIS2']
    #pixarcsec = stealhdr['CDELT2'] * 3600

    numpix = data_plot.shape[0]
    pixarcsec = data_wcs.wcs.cdelt[1]*3600

    args_plot = (numpix, pixarcsec)
    #dxy_arcsec = dxy * 206265
    #args_plot = (nxy, dxy_arcsec)
    logging.info(f'Parameters for model WCS: {args_plot}')

    logging.info('Fittype: %s', fittype)
    ring_model = model_prof(pars_bf, args_plot, vis_x_dat, 'plot', fittype)
    logging.info('Successfully computed model for plotting')

    phase_center = SkyCoord('10h43m57.7330s', '+11d42m12.9996s', frame='icrs')

    mod_wcs = make_model_wcs(phase_center.ra.deg, phase_center.dec.deg, pixarcsec, shape=(numpix, numpix))
    logging.info(mod_wcs)

    # Convert surface brightness from Jy/sr to MJy/sr (1 MJy = 10^6 Jy)
    data_plot /= 1e6
    resid_plot /= 1e6
    ring_model /= 1e6

    # --- Figure 1: Side-by-Side Comparison (Data, Model, Residuals) ---
    fig = plt.figure(figsize=(24, 8))

    ax1 = plt.subplot(131, projection=data_wcs)
    im1 = ax1.imshow(data_plot, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    ax1.text(5, 5, 'CLEAN Image', color='w', fontsize=16)
    cbar1 = plt.colorbar(mappable=im1, ax=ax1, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax2 = plt.subplot(132, projection=mod_wcs)
    im2 = ax2.imshow(ring_model, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    ax2.text(5, 5, 'Model Sky Intensity', color='w', fontsize=16)
    cbar2 = plt.colorbar(mappable=im2, ax=ax2, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax3 = plt.subplot(133, projection=data_wcs)
    im3 = ax3.imshow(resid_plot, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    ax3.text(5, 5, 'CLEAN Residual Visibilities', color='black', fontsize=16)
    cbar3 = plt.colorbar(mappable=im3, ax=ax3, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    # Format celestial axis labels
    axes = [ax1, ax2, ax3]
    for ax in axes:
        if ax != ax1:
            ax.set_ylabel(r'', size=0)
            ax.coords[1].set_ticklabel(size=0)
        else:
            ax.set_ylabel(r"Declination (J2000)", size=22, labelpad=1)
        if ax != ax2:
            ax.set_xlabel(r'', size=0)
        else:
            ax.set_xlabel(r"Right Ascension (J2000)", size=22)

    fig.suptitle("93GHZ Continuum Intensity of NGC 3351 (MJy/sr)", size=30)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    pp.savefig()

    # --- Figure 2: Comparison with Model Contour Overlays ---
    fig = plt.figure(figsize=(24, 8))

    ax1 = plt.subplot(131, projection=data_wcs)
    im1 = ax1.imshow(data_plot, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    ax1.text(5, 5, 'CLEAN Image', color='w', fontsize=16)
    cbar1 = plt.colorbar(mappable=im1, ax=ax1, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax2 = plt.subplot(132, projection=mod_wcs)
    im2 = ax2.imshow(ring_model, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    ax2.text(5, 5, 'Model Sky Intensity', color='w', fontsize=16)
    cbar2 = plt.colorbar(mappable=im2, ax=ax2, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax3 = plt.subplot(133, projection=data_wcs)
    im3 = ax3.imshow(resid_plot, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    ax3.text(5, 5, 'CLEAN Residual Visibilities', color='black', fontsize=16)
    cbar3 = plt.colorbar(mappable=im3, ax=ax3, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    axes = [ax1, ax2, ax3]

    for ax in axes:
        ax.set_autoscale_on(False)

    peak_flux = np.percentile(data_plot, 99.95)

    # Overlay model brightness contours at 20%, 40%, 60%, 80%, and 95% of peak intensity
    ax1.contour(ring_model, colors='w', transform=ax1.get_transform(mod_wcs), levels=peak_flux * np.array([0.2, 0.4, 0.6, 0.8, 0.95]), zorder=10, linewidths=0.5)
    ax2.contour(ring_model, colors='k', transform=ax2.get_transform(mod_wcs), levels=peak_flux * np.array([0.2, 0.4, 0.6, 0.8, 0.95]), zorder=10, linewidths=0.5)
    ax3.contour(ring_model, colors='w', transform=ax3.get_transform(mod_wcs), levels=peak_flux * np.array([0.2, 0.4, 0.6, 0.8, 0.95]), zorder=10, linewidths=0.5)

    for ax in axes:
        if ax != ax1:
            ax.set_ylabel(r'', size=0)
            ax.coords[1].set_ticklabel(size=0)
        else:
            ax.set_ylabel(r"Declination (J2000)", size=22, labelpad=1)
        if ax != ax2:
            ax.set_xlabel(r'', size=0)
        else:
            ax.set_xlabel(r"Right Ascension (J2000)", size=22)

    fig.suptitle("93GHZ Continuum Intensity of NGC 3351 (MJy/sr)", size=30)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    pp.savefig()

    # --- Figure 3: Comparison with Original Cluster Positions from Sun et al. 2024 ---
    fig = plt.figure(figsize=(24, 8))

    ax1 = plt.subplot(131, projection=data_wcs)
    im1 = ax1.imshow(data_plot, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    ax1.text(5, 5, 'CLEAN Image', color='w', fontsize=16)
    cbar1 = plt.colorbar(mappable=im1, ax=ax1, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax2 = plt.subplot(132, projection=mod_wcs)
    im2 = ax2.imshow(ring_model, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    ax2.text(5, 5, 'Model Sky Intensity', color='w', fontsize=16)
    cbar2 = plt.colorbar(mappable=im2, ax=ax2, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax3 = plt.subplot(133, projection=data_wcs)
    im3 = ax3.imshow(resid_plot, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    ax3.text(5, 5, 'CLEAN Residual Visibilities', color='black', fontsize=16)
    cbar3 = plt.colorbar(mappable=im3, ax=ax3, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ymc_ids = [15, 6, 18] #ideally this won't be hardcoded
    sun_coords = sun_radec(ymc_ids)
    dynest_coords = dynest_radec(pars_bf, fittype)

    sun_ra, sun_dec = zip(*sun_coords)
    dynest_ra, dynest_dec = zip(*dynest_coords)

    axes = [ax1, ax2, ax3]

    ax1.scatter(461.0, 460.0, marker='x', s=200)
    ax2.scatter(460.0, 460.0, marker='x', s=200)
    ax3.scatter(461.0, 460.0, marker='x', s=200)

    for ax in axes:
        ax.scatter(sun_ra, sun_dec, transform=ax.get_transform('world'), color='red', marker='x', s=120, linewidth=2)
        ax.scatter(dynest_ra, dynest_dec, transform=ax.get_transform('world'), color='blue', marker='x', s=120, linewidth=2)
        #ax.scatter(ngc3351.ra.deg, ngc3351.dec.deg, transform=ax.get_transform('world'), color='yellow', marker='x', s=120, linewidth=2)
        ax.scatter(phase_center.ra.deg, phase_center.dec.deg, transform=ax.get_transform('world'), color='yellow', marker='x', s=120, linewidth=2)


    for ax in axes:
        if ax != ax1:
            ax.set_ylabel(r'', size=0)
            ax.coords[1].set_ticklabel(size=0)
        else:
            ax.set_ylabel(r"Declination (J2000)", size=22, labelpad=1)
        if ax != ax2:
            ax.set_xlabel(r'', size=0)
        else:
            ax.set_xlabel(r"Right Ascension (J2000)", size=22)

    fig.suptitle("93GHZ Continuum Intensity of NGC 3351 (MJy/sr)", size=30)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    pp.savefig()

    pp.close()

    logging.info('Done!')

if __name__ == '__main__':
    main()