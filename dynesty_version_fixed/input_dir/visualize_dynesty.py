"""
Dynesty Visualization, Residual Imaging, and Post-Processing
=============================================================
This script post-processes the results of a Dynesty nested sampling run:
1. Restores the sampling state from the checkpoint file (`<fittype>_checkpoint.save`).
2. Extracts parameter posteriors and selects the maximum-likelihood joint sample
   as the best-fit model ($\hat{\theta}_{\text{ML}}$).
3. Generates Dynesty diagnostic figures (run progression summary and corner plot).
4. Evaluates best-fit synthetic visibilities and subtracts them from the observed data
   to produce residual visibilities: $V_{\text{resid}} = V_{\text{obs}} - V_{\text{mod}}$.
5. Implants residual visibilities into CASA Measurement Sets (`.ms`) and performs
   synthesis imaging (`tclean` with `niter=0` for dirty residuals) and primary beam correction.
6. Convolves the intrinsic sky model with the synthesized restoring beam for like-for-like
   image-plane comparison against the restored CLEAN data image.
7. Generates multi-panel comparison figures showing:
   [Observed CLEAN Image | Model Sky Intensity (Beam Convolved) | Residual Visibilities]
   with and without contour overlays, saving all figures to a multipage PDF.

Requirements:
    - Python environment with Dynesty, CASA (casatools, casatasks), Astropy, and Matplotlib.
"""

import os
os.environ["CASACORE_MEASURES_AUTO_UPDATE"] = "false"

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel, convolve_fft
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
    hdr = fits.open(infile)[0].header
    bmaj = hdr['BMAJ']  # Major axis FWHM in degrees
    bmin = hdr['BMIN']  # Minor axis FWHM in degrees

    # Gaussian beam solid angle in square degrees: pi / (4 * ln(2)) * FWHM_maj * FWHM_min
    omega_bm_deg2 = (np.pi / (4 * np.log(2))) * bmaj * bmin
    # Convert square degrees to steradians: 1 sr = (180 / pi)^2 deg^2
    omega_bm_sr = omega_bm_deg2 / (180 / np.pi)**2
    return omega_bm_sr

def img_prepper(fitsimg):
    """
    Loads a FITS image, extracts WCS coordinates, converts units, and crops a 2D cutout.

    Extracts a field of view centered on the galactic nucleus of NGC 3351.

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
    center = SkyCoord('10h43m57.7330s', '+11d42m12.9996s', frame='icrs')
    box_bkg = [36 * u.arcsecond, 36 * u.arcsecond]

    im = fits.open(fitsimg)[0]
    im_wcs = WCS(im.header, naxis=2)
    logging.info(im_wcs)
    conv = jybm_to_jysr(infile=fitsimg)
    # Convert image from Jy/beam to Jy/sr and crop around galaxy center
    im_plot = Cutout2D(im.data / conv, center, box_bkg, wcs=im_wcs)

    return im_plot.wcs, im_plot.data

def convolve_model_with_beam(model_2d, bmaj_deg, bmin_deg, bpa_deg, pixarcsec):
    """
    Convolves an intrinsic 2D sky brightness model with the synthesis restoring beam.

    Parameters
    ----------
    model_2d : numpy.ndarray
        2D intrinsic model image in surface brightness units.
    bmaj_deg : float
        Clean beam major axis FWHM in degrees.
    bmin_deg : float
        Clean beam minor axis FWHM in degrees.
    bpa_deg : float
        Clean beam position angle in degrees East of North.
    pixarcsec : float
        Pixel scale in arcseconds per pixel.

    Returns
    -------
    numpy.ndarray
        Beam-convolved model image matching the resolution of the restored CLEAN image.
    """
    if bmaj_deg <= 0 or bmin_deg <= 0:
        return model_2d

    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_maj_pix = (bmaj_deg * 3600.0 * fwhm_to_sigma) / pixarcsec
    sigma_min_pix = (bmin_deg * 3600.0 * fwhm_to_sigma) / pixarcsec

    # Astropy Gaussian2DKernel theta is measured counter-clockwise from X-axis
    theta_rad = np.radians(bpa_deg + 90.0)
    beam_kernel = Gaussian2DKernel(
        x_stddev=sigma_min_pix,
        y_stddev=sigma_maj_pix,
        theta=theta_rad
    )
    convolved = convolve_fft(model_2d, beam_kernel, normalize_kernel=True)
    return convolved

def circular_mean_and_dispersion(samples, weights, period=360.0):
    """
    Computes circular mean and angular dispersion for periodic parameters.
    period = 180.0 for axial parameters (e.g. disk PA); period = 360.0 for directional angles.
    """
    scale = 2.0 * np.pi / period
    s = np.sum(weights * np.sin(scale * samples))
    c = np.sum(weights * np.cos(scale * samples))
    w_sum = np.sum(weights)
    mean_angle = (np.arctan2(s, c) / scale) % period
    R = np.clip(np.hypot(s, c) / w_sum, 1e-12, 1.0)
    circ_std = np.sqrt(-2.0 * np.log(R)) / scale
    return mean_angle, circ_std

def main():
    """
    Main visualization and post-processing routine.
    """
    parser = argparse.ArgumentParser(
        description="Visualize Dynesty results, compute visibility residuals, and generate CASA images."
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
    # 1. Restore Dynesty Checkpoint & Coherent Best-Fit Parameter Selection
    # --------------------------------------------------------------------------
    check_file = './output/' + fittype + '_checkpoint.save'
    sampler = DynamicNestedSampler.restore(check_file)
    res = sampler.results

    logging.info(f"Read in the results from the saved checkpoint file {check_file}")

    labels, units, ndim = model_addon(fittype)

    logging.info(f"Dimensions for {fittype}: {ndim}")
    logging.info(f"Parameters for {fittype}: {labels}")

    logz = res.logz[-1]
    weights = np.exp(res.logwt - logz)

    # Use the maximum-likelihood posterior sample as the coherent joint best fit
    best_idx = np.argmax(res.logl)
    pars_bf = res.samples[best_idx].copy()
    logging.info(f"Coherent Joint Best-Fit (Maximum Likelihood) Parameters: {pars_bf}")

    # Also log marginal medians, credible intervals, and circular statistics for reporting
    logging.info("Marginal posterior medians and 68% credible intervals:")
    for i in range(ndim):
        label_lower = labels[i].lower()
        if 'posangle' in label_lower or (i == 4):
            c_mean, c_std = circular_mean_and_dispersion(res.samples[:, i], weights, period=180.0)
            logging.info(f"  {labels[i]:<25} : Circular Axial Mean = {c_mean:.4f} deg (circ std = {c_std:.4f} deg) [180-deg axial]")
        elif 'ang' in label_lower:
            c_mean, c_std = circular_mean_and_dispersion(res.samples[:, i], weights, period=360.0)
            logging.info(f"  {labels[i]:<25} : Circular Mean = {c_mean:.4f} deg (circ std = {c_std:.4f} deg) [360-deg circular]")

        quantiles = dyfunc.quantile(res.samples[:, i], [0.159, 0.5, 0.841], weights=weights)
        median = quantiles[1]
        m1 = median - quantiles[0]
        p1 = quantiles[2] - median
        if 'flux' in label_lower:
            flux_mjy = (10.0**median) * 1e3
            logging.info(f"  {labels[i]:<25} : {median:.4f} (+{p1:.4f} / -{m1:.4f}) {units[i]}  -->  Flux = {flux_mjy:.4f} mJy")
        elif 'sigma' in label_lower:
            sigma_arcsec = 10.0**median
            fwhm_arcsec = 2.35482 * sigma_arcsec
            logging.info(f"  {labels[i]:<25} : {median:.4f} (+{p1:.4f} / -{m1:.4f}) {units[i]}  -->  Sigma = {sigma_arcsec:.4f}\", FWHM = {fwhm_arcsec:.4f}\"")
        else:
            logging.info(f"  {labels[i]:<25} : {median:.4f} (+{p1:.4f} / -{m1:.4f}) {units[i]}")

    pdf_plot = './output/' + fittype + '_plots.pdf'
    pp = PdfPages(pdf_plot)
    logging.info(f'Plot File: {pdf_plot}')

    # --------------------------------------------------------------------------
    # 2. Dynesty Run Summary & Corner Plots
    # --------------------------------------------------------------------------
    fig, axes = dyplot.runplot(res)
    fig.tight_layout()
    pp.savefig(fig)
    plt.close(fig)
    logging.info('Successfully saved the summary plot...')

    logging.info(f"Total samples: {len(weights)}")
    logging.info(f"Non-zero weight samples: {np.count_nonzero(weights)}")
    logging.info(f"Max weight / Sum weight: {np.max(weights) / np.sum(weights):.4f}")
        
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
    ms = './casa_dir_simgauss/M95_C5+C2_cont93_uvtable.ms'
    msx = './casa_dir_simgauss/M95_C5+C2_cont93_trimmmedXX.ms'
    msy = './casa_dir_simgauss/M95_C5+C2_cont93_trimmmedYY.ms'

    data_imgname = './casa_dir_simgauss/M95_cont93GHz_auto'
    resid_imgname = './casa_dir_simgauss/M95_cont93GHz_residual'

    # Ingest XX baseline visibilities with frequency awareness
    raw_x = np.require(np.loadtxt(msx + '.uvtable.txt', unpack=True), requirements='C')
    if len(raw_x) >= 6:
        u_datx, v_datx, Re_datx, Im_datx, w_datx, freq_x = raw_x[:6]
        wavelength_x = 299792458.0 / freq_x
        u_datx = u_datx / wavelength_x
        v_datx = v_datx / wavelength_x
    else:
        u_datx, v_datx, Re_datx, Im_datx, w_datx = raw_x[:5]
    vis_datx = np.array(Re_datx + 1j * Im_datx, dtype=np.complex256)

    # Ingest YY baseline visibilities with frequency awareness
    raw_y = np.require(np.loadtxt(msy + '.uvtable.txt', unpack=True), requirements='C')
    if len(raw_y) >= 6:
        u_daty, v_daty, Re_daty, Im_daty, w_daty, freq_y = raw_y[:6]
        wavelength_y = 299792458.0 / freq_y
        u_daty = u_daty / wavelength_y
        v_daty = v_daty / wavelength_y
    else:
        u_daty, v_daty, Re_daty, Im_daty, w_daty = raw_y[:5]
    vis_daty = np.array(Re_daty + 1j * Im_daty, dtype=np.complex256)

    # Compute Fourier grid dimensions
    nxy, dxy = get_image_size(u_datx, v_datx, verbose=True)
    dxy_arcsec = dxy * 206265.0
    args_vis = (nxy, dxy)
    vis_x_dat = (u_datx, v_datx, Re_datx, Im_datx, w_datx)

    # Sample best-fit model visibilities using Galario
    logging.info('Evaluating best-fit model visibilities via Galario...')
    model = model_prof(pars_bf, args_vis, vis_x_dat, 'vis', fittype)
    logging.info('Successfully computed best fit model visibilities...')

    # Compute residual complex visibilities
    resid_visx = vis_datx - model
    resid_visy = vis_daty - model
    logging.info('Residual visibilities calculated...')

    # --------------------------------------------------------------------------
    # 3b. Visibility-Domain Residual Diagnostics (P2 Action Item)
    # --------------------------------------------------------------------------
    uvdist_x = np.hypot(u_datx, v_datx)
    chi2_vis_x = np.sum(w_datx * (np.real(resid_visx)**2 + np.imag(resid_visx)**2))
    dof_x = 2 * len(vis_datx) - ndim
    red_chi2_x = chi2_vis_x / dof_x
    logging.info(f"Visibility-domain XX fit: chi2 = {chi2_vis_x:.2e}, dof = {dof_x}, reduced chi2 = {red_chi2_x:.4f}")

    # Bin complex visibilities radially vs UV distance
    nbins = 30
    bins = np.linspace(np.min(uvdist_x), np.max(uvdist_x), nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_indices = np.digitize(uvdist_x, bins) - 1

    dat_re_binned = np.zeros(nbins)
    dat_re_err = np.zeros(nbins)
    mod_re_binned = np.zeros(nbins)
    res_re_binned = np.zeros(nbins)
    res_re_err = np.zeros(nbins)
    res_im_binned = np.zeros(nbins)
    res_im_err = np.zeros(nbins)

    for b in range(nbins):
        idx = (bin_indices == b)
        if np.any(idx):
            w_bin = w_datx[idx]
            sum_w = np.sum(w_bin)
            dat_re_binned[b] = np.sum(w_bin * np.real(vis_datx[idx])) / sum_w
            dat_re_err[b] = 1.0 / np.sqrt(sum_w)
            mod_re_binned[b] = np.sum(w_bin * np.real(model[idx])) / sum_w
            res_re_binned[b] = np.sum(w_bin * np.real(resid_visx[idx])) / sum_w
            res_re_err[b] = 1.0 / np.sqrt(sum_w)
            res_im_binned[b] = np.sum(w_bin * np.imag(resid_visx[idx])) / sum_w
            res_im_err[b] = 1.0 / np.sqrt(sum_w)

    valid_bins = (dat_re_err > 0)

    # Plot Visibility Diagnostics Figure
    fig_vis, (ax_v1, ax_v2, ax_v3) = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

    # Panel 1: Real visibilities vs UV distance (klambda)
    uv_klambda = bin_centers[valid_bins] / 1e3
    ax_v1.errorbar(uv_klambda, dat_re_binned[valid_bins] * 1e3, yerr=dat_re_err[valid_bins] * 1e3,
                   fmt='o', color='black', label='Observed Data (XX)', alpha=0.7, markersize=4)
    ax_v1.plot(uv_klambda, mod_re_binned[valid_bins] * 1e3, color='crimson', lw=2, label='Model Best Fit')
    ax_v1.set_ylabel('Real Visibility [mJy]')
    ax_v1.set_title(f'Visibility-Domain Diagnostic: {fittype} (reduced $\\chi^2 = {red_chi2_x:.2f}$)')
    ax_v1.legend(loc='upper right')
    ax_v1.grid(True, linestyle=':', alpha=0.6)

    # Panel 2: Complex residuals vs UV distance
    ax_v2.errorbar(uv_klambda, res_re_binned[valid_bins] * 1e3, yerr=res_re_err[valid_bins] * 1e3,
                   fmt='s', color='dodgerblue', label='Real Residual', alpha=0.7, markersize=4)
    ax_v2.errorbar(uv_klambda, res_im_binned[valid_bins] * 1e3, yerr=res_im_err[valid_bins] * 1e3,
                   fmt='^', color='orange', label='Imag Residual', alpha=0.7, markersize=4)
    ax_v2.axhline(0, color='gray', linestyle='--')
    ax_v2.set_xlabel(r'UV Distance [$k\lambda$]')
    ax_v2.set_ylabel('Residual Visibility [mJy]')
    ax_v2.legend(loc='upper right')
    ax_v2.grid(True, linestyle=':', alpha=0.6)

    # Panel 3: Standardized residuals histogram vs standard normal N(0, 1)
    sub_sample_size = min(len(resid_visx), 200000)
    sub_idx = np.random.choice(len(resid_visx), size=sub_sample_size, replace=False)
    norm_res_re = np.real(resid_visx[sub_idx]) * np.sqrt(w_datx[sub_idx])
    norm_res_im = np.imag(resid_visx[sub_idx]) * np.sqrt(w_datx[sub_idx])
    norm_res = np.concatenate([norm_res_re, norm_res_im])

    ax_v3.hist(norm_res, bins=100, range=(-5, 5), density=True,
               alpha=0.6, color='steelblue', label=r'Residuals $(V_{\rm res} \cdot \sqrt{w})$')
    x_gauss = np.linspace(-5, 5, 200)
    ax_v3.plot(x_gauss, np.exp(-0.5 * x_gauss**2) / np.sqrt(2 * np.pi), 'r-', lw=2, label=r'Standard Normal $\mathcal{N}(0, 1)$')
    ax_v3.set_xlabel(r'Standardized Residual ($\sigma$)')
    ax_v3.set_ylabel('Probability Density')
    ax_v3.legend(loc='upper right')
    ax_v3.grid(True, linestyle=':', alpha=0.6)

    fig_vis.tight_layout()
    pp.savefig(fig_vis)
    plt.close(fig_vis)
    logging.info('Successfully saved visibility-domain diagnostics plot...')

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
    tb.flush()
    tb.close()

    # Write residuals for YY polarization
    os.system('rm -rf ' + msy + '.residual.ms')
    os.system('cp -R ' + msy + ' ' + msy + '.residual.ms')
    tb.open(msy + '.residual.ms', nomodify=False)
    tb.putcol('DATA', implant_visy)
    tb.flush()
    tb.close()

    logging.info('Residual data put into .residual.ms')

    # --------------------------------------------------------------------------
    # 5. CASA Synthesis Imaging and Primary Beam Correction
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

    # Image residual visibilities: use niter=0 for pristine dirty residual diagnostic
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
        niter=0,  # Dirty residual map: prevents non-linear deconvolution of noise
        interactive=False
    )

    # Primary beam correction and FITS export
    os.system('rm -rf ' + data_imgname + '.image.pbcor')
    os.system('rm -rf ' + data_imgname + '.fits')
    os.system('rm -rf ' + data_imgname + '.image.fits')
    os.system('rm -rf ' + resid_imgname + '.image.pbcor')
    os.system('rm -rf ' + resid_imgname + '.fits')
    os.system('rm -rf ' + resid_imgname + '.dirty.fits')

    # Export both uncorrected dirty residual (stationary noise) and PB-corrected images
    exportfits(data_imgname + '.image', fitsimage=data_imgname + '.image.fits', dropdeg=True)
    impbcor(data_imgname + '.image', pbimage=data_imgname + '.pb', outfile=data_imgname + '.image.pbcor')
    exportfits(data_imgname + '.image.pbcor', fitsimage=data_imgname + '.fits', dropdeg=True)

    exportfits(resid_imgname + '.image', fitsimage=resid_imgname + '.dirty.fits', dropdeg=True)
    impbcor(resid_imgname + '.image', pbimage=resid_imgname + '.pb', outfile=resid_imgname + '.image.pbcor')
    exportfits(resid_imgname + '.image.pbcor', fitsimage=resid_imgname + '.fits', dropdeg=True)

    logging.info('CASA imaging and FITS export completed...')

    # --------------------------------------------------------------------------
    # 6. Generate 2D Model Sky Map & Multi-Panel Comparison Figures
    # --------------------------------------------------------------------------
    data_wcs, data_plot = img_prepper(data_imgname + '.fits')
    resid_wcs, resid_plot = img_prepper(resid_imgname + '.fits')

    # Header parameters for restoring beam
    hdr_data = fits.open(data_imgname + '.fits')[0].header
    bmaj_deg = hdr_data.get('BMAJ', 0.0)
    bmin_deg = hdr_data.get('BMIN', 0.0)
    bpa_deg = hdr_data.get('BPA', 0.0)

    numpix = data_plot.shape[0]
    pixarcsec = abs(data_wcs.wcs.cdelt[1]) * 3600.0

    args_plot = (numpix, pixarcsec)
    logging.info(f'Parameters for model grid: {args_plot}')

    ring_model_intrinsic = model_prof(pars_bf, args_plot, vis_x_dat, 'plot', fittype)
    logging.info('Successfully computed intrinsic model for plotting...')

    # Convolve model with synthesized restoring beam for like-for-like comparison
    ring_model = convolve_model_with_beam(ring_model_intrinsic, bmaj_deg, bmin_deg, bpa_deg, pixarcsec)
    logging.info('Convolved intrinsic model with restoring beam...')

    phase_center = SkyCoord('10h43m57.7330s', '+11d42m12.9996s', frame='icrs')
    mod_wcs = make_model_wcs(phase_center.ra.deg, phase_center.dec.deg, pixarcsec, shape=(numpix, numpix))

    # Convert surface brightness from Jy/sr to MJy/sr
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
    ax2.text(5, 5, 'Model Sky Intensity (Beam Convolved)', color='w', fontsize=16)
    cbar2 = plt.colorbar(mappable=im2, ax=ax2, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax3 = plt.subplot(133, projection=data_wcs)
    im3 = ax3.imshow(resid_plot, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    ax3.text(5, 5, 'Dirty Residual Visibilities', color='black', fontsize=16)
    cbar3 = plt.colorbar(mappable=im3, ax=ax3, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

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

    fig.suptitle("93 GHz Continuum Intensity of NGC 3351 (MJy/sr)", size=30)
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
    ax2.text(5, 5, 'Model Sky Intensity (Beam Convolved)', color='w', fontsize=16)
    cbar2 = plt.colorbar(mappable=im2, ax=ax2, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax3 = plt.subplot(133, projection=data_wcs)
    im3 = ax3.imshow(resid_plot, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    ax3.text(5, 5, 'Dirty Residual Visibilities', color='black', fontsize=16)
    cbar3 = plt.colorbar(mappable=im3, ax=ax3, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    for ax in axes:
        ax.set_autoscale_on(False)

    peak_flux = np.percentile(data_plot, 99.95)

    # Overlay model brightness contours
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

    fig.suptitle("93 GHz Continuum Intensity of NGC 3351 (MJy/sr)", size=30)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    pp.savefig()

    # --- Figure 3: Comparison with Reference Cluster Positions ---
    fig = plt.figure(figsize=(24, 8))

    ax1 = plt.subplot(131, projection=data_wcs)
    im1 = ax1.imshow(data_plot, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    ax1.text(5, 5, 'CLEAN Image', color='w', fontsize=16)
    cbar1 = plt.colorbar(mappable=im1, ax=ax1, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax2 = plt.subplot(132, projection=mod_wcs)
    im2 = ax2.imshow(ring_model, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    ax2.text(5, 5, 'Model Sky Intensity (Beam Convolved)', color='w', fontsize=16)
    cbar2 = plt.colorbar(mappable=im2, ax=ax2, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax3 = plt.subplot(133, projection=data_wcs)
    im3 = ax3.imshow(resid_plot, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    ax3.text(5, 5, 'Dirty Residual Visibilities', color='black', fontsize=16)
    cbar3 = plt.colorbar(mappable=im3, ax=ax3, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ymc_ids = [15, 6, 18]
    sun_coords = sun_radec(ymc_ids)
    dynest_coords = dynest_radec(pars_bf, fittype)

    sun_ra, sun_dec = zip(*sun_coords)
    dynest_ra, dynest_dec = zip(*dynest_coords)

    for ax in axes:
        ax.scatter(sun_ra, sun_dec, transform=ax.get_transform('world'), color='red', marker='x', s=120, linewidth=2, label='Sun et al. (2024)')
        ax.scatter(dynest_ra, dynest_dec, transform=ax.get_transform('world'), color='blue', marker='x', s=120, linewidth=2, label='Fitted Model')
        ax.scatter(phase_center.ra.deg, phase_center.dec.deg, transform=ax.get_transform('world'), color='yellow', marker='+', s=120, linewidth=2, label='Phase Center')

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

    fig.suptitle("93 GHz Continuum Intensity of NGC 3351 (MJy/sr)", size=30)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    pp.savefig()

    # Save FITS products with full WCS information
    header_data = data_wcs.to_header()
    header_model = mod_wcs.to_header()

    hdu_data = fits.PrimaryHDU(data=data_plot, header=header_data)
    hdu_model = fits.PrimaryHDU(data=ring_model, header=header_model)
    hdu_resid = fits.PrimaryHDU(data=resid_plot, header=header_data)

    hdu_data.writeto("./output/data_plot.fits", overwrite=True)
    hdu_model.writeto("./output/ring_model.fits", overwrite=True)
    hdu_resid.writeto("./output/resid_plot.fits", overwrite=True)

    # --------------------------------------------------------------------------
    # 7. Zoomed-In Blob Analysis & 1D Brightness Profiles (Data, Model, Residuals)
    # --------------------------------------------------------------------------
    blob_box_size = [2 * u.arcsecond, 2 * u.arcsecond]

    for b_idx, (b_ra, b_dec) in enumerate(dynest_coords):
        blob_coord = SkyCoord(b_ra * u.deg, b_dec * u.deg, frame='icrs')

        cutout_data = Cutout2D(data_plot, blob_coord, blob_box_size, wcs=data_wcs)
        cutout_mod = Cutout2D(ring_model, blob_coord, blob_box_size, wcs=mod_wcs)
        cutout_res = Cutout2D(resid_plot, blob_coord, blob_box_size, wcs=data_wcs)

        cutouts = [
            ('Observed CLEAN Image', cutout_data),
            ('Model Sky Intensity', cutout_mod),
            ('Dirty Residual Visibilities', cutout_res)
        ]

        all_cut_data = [cut.data for _, cut in cutouts]
        vmin_zoom = min(0.0, min(np.percentile(d, 1) for d in all_cut_data))
        vmax_zoom = max(np.max(d) for d in all_cut_data)

        ylim_min = vmin_zoom - 0.02 * (vmax_zoom - vmin_zoom) if vmin_zoom < 0 else 0.0
        ylim_max = vmax_zoom * 1.05

        for target_name, cutout in cutouts:
            cut_data = cutout.data
            cut_wcs = cutout.wcs

            cx_f, cy_f = cut_wcs.world_to_pixel(blob_coord)
            cx = int(round(float(cx_f)))
            cy = int(round(float(cy_f)))

            # Bounds clamping to prevent IndexError and ensure 3-pixel slice stays within cutout
            ny_cut, nx_cut = cut_data.shape
            cy_clamp = int(np.clip(cy, 1, ny_cut - 2))
            cx_clamp = int(np.clip(cx, 1, nx_cut - 2))

            # 1D RA brightness profile
            ra_profile = np.mean(cut_data[cy_clamp - 1 : cy_clamp + 2, :], axis=0)
            x_indices = np.arange(nx_cut)
            pixel_world_ra = cut_wcs.pixel_to_world(x_indices, np.full(nx_cut, cy_clamp))
            ra_coords = pixel_world_ra.ra.deg

            # 1D Dec brightness profile
            dec_profile = np.mean(cut_data[:, cx_clamp - 1 : cx_clamp + 2], axis=1)
            y_indices = np.arange(ny_cut)
            pixel_world_dec = cut_wcs.pixel_to_world(np.full(ny_cut, cx_clamp), y_indices)
            dec_coords = pixel_world_dec.dec.deg

            # --- Layout: Left panel = 2D Zoomed Cutout; Right panels = 1D RA & Dec Profiles ---
            fig = plt.figure(figsize=(24, 8))

            ax_img = plt.subplot(131, projection=cut_wcs)
            im_zoom = ax_img.imshow(cut_data, vmin=vmin_zoom, vmax=vmax_zoom, origin='lower', cmap='inferno', rasterized=True)
            ax_img.text(5, 5, f'{target_name} (Zoom)', color='w', fontsize=16)
            cbar_zoom = plt.colorbar(mappable=im_zoom, ax=ax_img, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)
            cbar_zoom.set_label(r'Specific Intensity (MJy/sr)', size=16)

            # Central blob coordinate marker
            ax_img.scatter(b_ra, b_dec, transform=ax_img.get_transform('world'), color='cyan', marker='+', s=150, linewidth=2)

            # Slice indicators
            ax_img.axhline(cy_clamp - 1.5, color='crimson', linestyle=':', linewidth=1.5)
            ax_img.axhline(cy_clamp + 1.5, color='crimson', linestyle=':', linewidth=1.5)
            ax_img.axvline(cx_clamp - 1.5, color='dodgerblue', linestyle=':', linewidth=1.5)
            ax_img.axvline(cx_clamp + 1.5, color='dodgerblue', linestyle=':', linewidth=1.5)

            ax_img.set_xlabel(r"Right Ascension (J2000)", size=20)
            ax_img.set_ylabel(r"Declination (J2000)", size=20, labelpad=1)

            # Panel 2: 1D RA Profile
            ax_ra = plt.subplot(132)
            ax_ra.plot(ra_coords, ra_profile, color='crimson', lw=2.5)
            ax_ra.axvline(b_ra, color='gray', linestyle='--', alpha=0.7)
            ax_ra.set_xlim(ra_coords[0], ra_coords[-1])
            ax_ra.set_ylim(ylim_min, ylim_max)
            ax_ra.set_xlabel(r"Right Ascension (deg)", size=20)
            ax_ra.set_ylabel(r"Specific Intensity (MJy/sr)", size=20)
            ax_ra.set_title(r"1D RA Profile ($\pm 1$ Dec pixel avg)", size=18)
            ax_ra.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            formatter_ra = ticker.ScalarFormatter(useOffset=False)
            formatter_ra.set_scientific(False)
            ax_ra.xaxis.set_major_formatter(formatter_ra)
            plt.setp(ax_ra.get_xticklabels(), rotation=20, ha='right')
            ax_ra.grid(True, alpha=0.3, linestyle=':')

            # Panel 3: 1D Dec Profile
            ax_dec = plt.subplot(133)
            ax_dec.plot(dec_coords, dec_profile, color='dodgerblue', lw=2.5)
            ax_dec.axvline(b_dec, color='gray', linestyle='--', alpha=0.7)
            ax_dec.set_xlim(dec_coords[0], dec_coords[-1])
            ax_dec.set_ylim(ylim_min, ylim_max)
            ax_dec.set_xlabel(r"Declination (deg)", size=20)
            ax_dec.set_ylabel(r"Specific Intensity (MJy/sr)", size=20)
            ax_dec.set_title(r"1D Dec Profile ($\pm 1$ RA pixel avg)", size=18)
            ax_dec.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            formatter_dec = ticker.ScalarFormatter(useOffset=False)
            formatter_dec.set_scientific(False)
            ax_dec.xaxis.set_major_formatter(formatter_dec)
            plt.setp(ax_dec.get_xticklabels(), rotation=20, ha='right')
            ax_dec.grid(True, alpha=0.3, linestyle=':')

            blob_label = f"Peak {b_idx + 1}" if "2peak" in fittype else (f"Blob {b_idx + 1}" if len(dynest_coords) > 1 else "Blob")
            fig.suptitle(f"{target_name} - {blob_label} Zoom ($2'' \\times 2''$)", size=26)
            fig.subplots_adjust(hspace=0.2, wspace=0.28, bottom=0.15)
            pp.savefig()
            plt.close(fig)

    pp.close()
    logging.info('Post-processing and figure generation finished successfully!')

if __name__ == '__main__':
    main()
