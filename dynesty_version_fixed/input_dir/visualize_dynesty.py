r"""
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
from astropy.visualization.wcsaxes import add_beam
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
import prior_tform_streamline as pt_stream
from uvplot import UVTable
from matplotlib.backends.backend_pdf import PdfPages
import logging
import corner

import shutil
# Enable LaTeX typesetting if latex is available on the host system
if shutil.which('latex'):
    rc('text', usetex=True)
else:
    rc('text', usetex=False)
font = {'family': 'serif',
        'weight': 'bold',
        'size': '14'}
rc('font', **font)

def jybm_to_jysr(infile):
    r"""
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

    # Astropy Gaussian2DKernel theta is measured counter-clockwise from X-axis.
    # At theta = 0, y_stddev (sigma_maj) is aligned with Y-axis (North).
    # Since BPA is measured East of North (counter-clockwise on the sky), theta = bpa_deg.
    theta_rad = np.radians(bpa_deg)
    beam_kernel = Gaussian2DKernel(
        x_stddev=sigma_min_pix,
        y_stddev=sigma_maj_pix,
        theta=theta_rad
    )
    convolved = convolve_fft(model_2d, beam_kernel, normalize_kernel=True)
    return convolved

def safe_add_beam(ax, header=None, frame=False, color='white', corner='bottom left'):
    """
    Safely adds a synthesised restoring beam patch to a celestial WCSAxes subplot.
    """
    if header is not None and 'BMAJ' in header:
        try:
            add_beam(ax, header=header, frame=frame, color=color, corner=corner)
        except Exception as e:
            logging.warning(f"Could not add beam to axis: {e}")

def render_summary_table_page(res, pars_bf, weights, fittype, pp):
    """
    Constructs and saves the final summary table PDF page containing:
    - Free parameters fitted by Dynesty
    - Physical conversions:
        - Ring Flux (Jy) [after Ring LogFlux]
        - Ring Peak (Jy/sr) [after Ring Flux]
        - Ring Sigma (arcsec) [after Ring LogSigma]
        - Ring FWHM (arcsec) [after Ring Sigma]
        - Equivalent Flux, Peak, Sigma, and FWHM for every modeled blob/clump component.
    - Columns: Prior Low, Prior High, ML Estimate, 50th quantile, 16th quantile, 84th quantile.
    """
    ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
    FWHM_FACTOR = 2.354820045

    def get_quantiles(arr):
        return dyfunc.quantile(np.asarray(arr), [0.16, 0.50, 0.84], weights=weights)

    def fmt_cell(val):
        if val is None or np.isnan(val):
            return "-"
        abs_v = abs(val)
        if abs_v == 0.0:
            return "0.0"
        if abs_v >= 1e4 or abs_v < 1e-2:
            return f"{val:.3e}"
        elif abs_v >= 100.0:
            return f"{val:.2f}"
        else:
            return f"{val:.3f}"

    # List of table rows: [Parameter, Unit, Prior Low, Prior High, ML, 50th, 16th, 84th]
    table_rows = []

    def add_table_row(name, unit, p_low, p_high, ml_val, q_vals):
        table_rows.append([
            name,
            unit,
            fmt_cell(p_low),
            fmt_cell(p_high),
            fmt_cell(ml_val),
            fmt_cell(q_vals[1]),
            fmt_cell(q_vals[0]),
            fmt_cell(q_vals[2])
        ])

    # 1. Ring Component (if present in model)
    if fittype not in ('simgauss', 'twod_simgauss'):
        # Row 1: Ring LogFlux
        q_lf = get_quantiles(res.samples[:, 0])
        add_table_row("Ring LogFlux", "log(Jy)", pt_stream.RING_PRIOR_RANGES[0, 0], pt_stream.RING_PRIOR_RANGES[0, 1],
                      pars_bf[0], q_lf)

        # Row 2: Ring Flux (Jy)
        flux_samples = 10.0**res.samples[:, 0]
        q_flux = get_quantiles(flux_samples)
        add_table_row("Ring Flux", "Jy", 10.0**pt_stream.RING_PRIOR_RANGES[0, 0], 10.0**pt_stream.RING_PRIOR_RANGES[0, 1],
                      10.0**pars_bf[0], q_flux)

        # Row 3: Ring Peak (Jy/sr)
        cos_inc_samples = np.maximum(np.cos(np.radians(res.samples[:, 3])), 0.05)
        area_ring_samples = ((2.0 * np.pi)**1.5) * (res.samples[:, 2] * ARCSEC_TO_RAD) * (10.0**res.samples[:, 1] * ARCSEC_TO_RAD) * cos_inc_samples
        ring_peak_samples = flux_samples / np.maximum(area_ring_samples, 1e-30)
        q_rpeak = get_quantiles(ring_peak_samples)

        cos_inc_ml = max(float(np.cos(np.radians(pars_bf[3]))), 0.05)
        area_ring_ml = ((2.0 * np.pi)**1.5) * (pars_bf[2] * ARCSEC_TO_RAD) * (10.0**pars_bf[1] * ARCSEC_TO_RAD) * cos_inc_ml
        ring_peak_ml = (10.0**pars_bf[0]) / max(area_ring_ml, 1e-30)

        add_table_row("Ring Peak", "Jy/sr", 10.0**pt_stream.COMMON_RING_PRIORS[0, 0], 10.0**pt_stream.COMMON_RING_PRIORS[0, 1],
                      ring_peak_ml, q_rpeak)

        # Row 4: Ring LogSigma
        q_ls = get_quantiles(res.samples[:, 1])
        add_table_row("Ring LogSigma", "log(arcsec)", pt_stream.RING_PRIOR_RANGES[1, 0], pt_stream.RING_PRIOR_RANGES[1, 1],
                      pars_bf[1], q_ls)

        # Row 5: Ring Sigma (arcsec)
        sigma_samples = 10.0**res.samples[:, 1]
        q_sigma = get_quantiles(sigma_samples)
        add_table_row("Ring Sigma", "arcsec", pt_stream.COMMON_RING_PRIORS[1, 0], pt_stream.COMMON_RING_PRIORS[1, 1],
                      10.0**pars_bf[1], q_sigma)

        # Row 6: Ring FWHM (arcsec)
        fwhm_samples = FWHM_FACTOR * sigma_samples
        q_fwhm = get_quantiles(fwhm_samples)
        add_table_row("Ring FWHM", "arcsec", FWHM_FACTOR * pt_stream.COMMON_RING_PRIORS[1, 0], FWHM_FACTOR * pt_stream.COMMON_RING_PRIORS[1, 1],
                      FWHM_FACTOR * 10.0**pars_bf[1], q_fwhm)

        # Rows 7-11: Remaining ring parameters
        ring_param_names = ["Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec"]
        ring_param_units = ["arcsec", "degrees", "degrees", "arcsec", "arcsec"]
        for p_idx, (p_name, p_unit) in enumerate(zip(ring_param_names, ring_param_units), start=2):
            q_p = get_quantiles(res.samples[:, p_idx])
            add_table_row(p_name, p_unit, pt_stream.COMMON_RING_PRIORS[p_idx, 0], pt_stream.COMMON_RING_PRIORS[p_idx, 1],
                          pars_bf[p_idx], q_p)

    # Helper function for adding a blob component
    def add_blob_component(blob_label, idx_flux, idx_sigma, prior_logflux_range, prior_peak_log_range, prior_sigma_range):
        # 1. LogFlux
        q_lf = get_quantiles(res.samples[:, idx_flux])
        add_table_row(f"{blob_label} LogFlux", "log(Jy)", prior_logflux_range[0], prior_logflux_range[1],
                      pars_bf[idx_flux], q_lf)

        # 2. Flux (Jy)
        b_flux_samples = 10.0**res.samples[:, idx_flux]
        q_bflux = get_quantiles(b_flux_samples)
        add_table_row(f"{blob_label} Flux", "Jy", 10.0**prior_logflux_range[0], 10.0**prior_logflux_range[1],
                      10.0**pars_bf[idx_flux], q_bflux)

        # 3. Peak (Jy/sr)
        b_area_samples = 2.0 * np.pi * ((10.0**res.samples[:, idx_sigma] * ARCSEC_TO_RAD)**2)
        b_peak_samples = b_flux_samples / np.maximum(b_area_samples, 1e-30)
        q_bpeak = get_quantiles(b_peak_samples)

        b_area_ml = 2.0 * np.pi * ((10.0**pars_bf[idx_sigma] * ARCSEC_TO_RAD)**2)
        b_peak_ml = (10.0**pars_bf[idx_flux]) / max(b_area_ml, 1e-30)
        add_table_row(f"{blob_label} Peak", "Jy/sr", 10.0**prior_peak_log_range[0], 10.0**prior_peak_log_range[1],
                      b_peak_ml, q_bpeak)

        # 4. LogSigma
        prior_logsigma_min = np.log10(prior_sigma_range[0])
        prior_logsigma_max = np.log10(prior_sigma_range[1])
        q_ls = get_quantiles(res.samples[:, idx_sigma])
        add_table_row(f"{blob_label} LogSigma", "log(arcsec)", prior_logsigma_min, prior_logsigma_max,
                      pars_bf[idx_sigma], q_ls)

        # 5. Sigma (arcsec)
        b_sigma_samples = 10.0**res.samples[:, idx_sigma]
        q_bsigma = get_quantiles(b_sigma_samples)
        add_table_row(f"{blob_label} Sigma", "arcsec", prior_sigma_range[0], prior_sigma_range[1],
                      10.0**pars_bf[idx_sigma], q_bsigma)

        # 6. FWHM (arcsec)
        b_fwhm_samples = FWHM_FACTOR * b_sigma_samples
        q_bfwhm = get_quantiles(b_fwhm_samples)
        add_table_row(f"{blob_label} FWHM", "arcsec", FWHM_FACTOR * prior_sigma_range[0], FWHM_FACTOR * prior_sigma_range[1],
                      FWHM_FACTOR * 10.0**pars_bf[idx_sigma], q_bfwhm)

    # 2. Model-Specific Blob Components
    if fittype == 'twod_gauss1blob':
        add_blob_component("B1", 7, 8,
                           pt_stream.GAUSS1BLOB_PRIOR_RANGES[7],
                           pt_stream.GAUSS1BLOB_USER_PRIORS[0],
                           pt_stream.GAUSS1BLOB_USER_PRIORS[1])
        q_dist = get_quantiles(res.samples[:, 9])
        add_table_row("B1 Dist", "arcsec", pt_stream.GAUSS1BLOB_USER_PRIORS[2, 0], pt_stream.GAUSS1BLOB_USER_PRIORS[2, 1],
                      pars_bf[9], q_dist)
        q_ang = get_quantiles(res.samples[:, 10])
        add_table_row("B1 Angle", "degrees", pt_stream.GAUSS1BLOB_USER_PRIORS[3, 0], pt_stream.GAUSS1BLOB_USER_PRIORS[3, 1],
                      pars_bf[10], q_ang)

    elif fittype == 'twod_gauss1blob_2peak':
        # Core
        add_blob_component("B11 (Core)", 7, 8,
                           pt_stream.GAUSS1BLOB_2PEAK_PRIOR_RANGES[7],
                           pt_stream.GAUSS1BLOB_2PEAK_USER_PRIORS[0],
                           pt_stream.GAUSS1BLOB_2PEAK_USER_PRIORS[1])
        # Envelope
        add_blob_component("B12 (Env)", 9, 10,
                           pt_stream.GAUSS1BLOB_2PEAK_PRIOR_RANGES[9],
                           pt_stream.GAUSS1BLOB_2PEAK_USER_PRIORS[2],
                           pt_stream.GAUSS1BLOB_2PEAK_USER_PRIORS[3])
        q_dist = get_quantiles(res.samples[:, 11])
        add_table_row("Dist", "arcsec", pt_stream.GAUSS1BLOB_2PEAK_USER_PRIORS[4, 0], pt_stream.GAUSS1BLOB_2PEAK_USER_PRIORS[4, 1],
                      pars_bf[11], q_dist)
        q_ang = get_quantiles(res.samples[:, 12])
        add_table_row("Angle", "degrees", pt_stream.GAUSS1BLOB_2PEAK_USER_PRIORS[5, 0], pt_stream.GAUSS1BLOB_2PEAK_USER_PRIORS[5, 1],
                      pars_bf[12], q_ang)

    elif fittype == 'twod_gauss1blob_2peak_dp':
        # Peak 1
        add_blob_component("B11 (Peak 1)", 7, 8,
                           pt_stream.GAUSS1BLOB_2PEAK_DP_PRIOR_RANGES[7],
                           pt_stream.GAUSS1BLOB_2PEAK_DP_USER_PRIORS[0],
                           pt_stream.GAUSS1BLOB_2PEAK_DP_USER_PRIORS[1])
        q_d1 = get_quantiles(res.samples[:, 9])
        add_table_row("Dist 1", "arcsec", pt_stream.GAUSS1BLOB_2PEAK_DP_USER_PRIORS[2, 0], pt_stream.GAUSS1BLOB_2PEAK_DP_USER_PRIORS[2, 1],
                      pars_bf[9], q_d1)
        q_a1 = get_quantiles(res.samples[:, 10])
        add_table_row("Angle 1", "degrees", pt_stream.GAUSS1BLOB_2PEAK_DP_USER_PRIORS[3, 0], pt_stream.GAUSS1BLOB_2PEAK_DP_USER_PRIORS[3, 1],
                      pars_bf[10], q_a1)

        # Peak 2
        add_blob_component("B12 (Peak 2)", 11, 12,
                           pt_stream.GAUSS1BLOB_2PEAK_DP_PRIOR_RANGES[11],
                           pt_stream.GAUSS1BLOB_2PEAK_DP_USER_PRIORS[4],
                           pt_stream.GAUSS1BLOB_2PEAK_DP_USER_PRIORS[5])
        q_d2 = get_quantiles(res.samples[:, 13])
        add_table_row("Dist 2", "arcsec", pt_stream.GAUSS1BLOB_2PEAK_DP_USER_PRIORS[6, 0], pt_stream.GAUSS1BLOB_2PEAK_DP_USER_PRIORS[6, 1],
                      pars_bf[13], q_d2)
        q_a2 = get_quantiles(res.samples[:, 14])
        add_table_row("Angle 2", "degrees", pt_stream.GAUSS1BLOB_2PEAK_DP_USER_PRIORS[7, 0], pt_stream.GAUSS1BLOB_2PEAK_DP_USER_PRIORS[7, 1],
                      pars_bf[14], q_a2)

    elif fittype == 'twod_gauss3blob':
        for k, name in [(1, 'B1'), (2, 'B2'), (3, 'B3')]:
            idx_base = 7 + (k - 1) * 4
            u_base = (k - 1) * 4
            add_blob_component(name, idx_base, idx_base + 1,
                               pt_stream.GAUSS3BLOB_PRIOR_RANGES[idx_base],
                               pt_stream.GAUSS3BLOB_USER_PRIORS[u_base],
                               pt_stream.GAUSS3BLOB_USER_PRIORS[u_base + 1])
            q_dist = get_quantiles(res.samples[:, idx_base + 2])
            add_table_row(f"{name} Dist", "arcsec", pt_stream.GAUSS3BLOB_USER_PRIORS[u_base + 2, 0], pt_stream.GAUSS3BLOB_USER_PRIORS[u_base + 2, 1],
                          pars_bf[idx_base + 2], q_dist)
            q_ang = get_quantiles(res.samples[:, idx_base + 3])
            add_table_row(f"{name} Angle", "degrees", pt_stream.GAUSS3BLOB_USER_PRIORS[u_base + 3, 0], pt_stream.GAUSS3BLOB_USER_PRIORS[u_base + 3, 1],
                          pars_bf[idx_base + 3], q_ang)

    elif fittype in ('simgauss', 'twod_simgauss'):
        add_blob_component("Blob", 0, 1,
                           pt_stream.SIMGAUSS_PRIOR_RANGES[0],
                           pt_stream.SIMGAUSS_USER_PRIORS[0],
                           pt_stream.SIMGAUSS_USER_PRIORS[1])
        sim_names = ["Dist", "Angle", "PA", "Offset RA", "Offset Dec"]
        sim_units = ["arcsec", "degrees", "degrees", "arcsec", "arcsec"]
        for s_idx, (s_name, s_unit) in enumerate(zip(sim_names, sim_units), start=2):
            q_s = get_quantiles(res.samples[:, s_idx])
            add_table_row(s_name, s_unit, pt_stream.SIMGAUSS_USER_PRIORS[s_idx, 0], pt_stream.SIMGAUSS_USER_PRIORS[s_idx, 1],
                          pars_bf[s_idx], q_s)

    # Render Table onto Matplotlib Figure
    col_labels = [
        "Parameter",
        "Unit",
        "Prior Low",
        "Prior High",
        "ML Estimate",
        "50th (Median)",
        "16th",
        "84th"
    ]

    fig_height = max(8.5, 0.40 * len(table_rows) + 2.5)
    fig_tab, ax_tab = plt.subplots(figsize=(14, fig_height))
    ax_tab.axis('off')

    tab = ax_tab.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1.05, 1.4)

    # Style header row with light gray background
    for col_idx in range(len(col_labels)):
        cell = tab[0, col_idx]
        cell.set_facecolor('#d9d9d9')
        cell.set_text_props(weight='bold')

    # Alternate row colors for enhanced readability
    for row_idx in range(1, len(table_rows) + 1):
        bg_color = '#f5f5f5' if row_idx % 2 == 0 else '#ffffff'
        for col_idx in range(len(col_labels)):
            tab[row_idx, col_idx].set_facecolor(bg_color)

    fig_tab.suptitle(f"Dynesty Posterior Summary & Derived Parameters: {fittype}", fontsize=16, y=0.98)
    fig_tab.tight_layout()
    pp.savefig(fig_tab, bbox_inches='tight')
    plt.close(fig_tab)
    logging.info('Successfully saved final summary table page to PDF...')

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
    try:
        fig, axes = dyplot.runplot(res, logplot=True)
    except Exception:
        fig, axes = dyplot.runplot(res)

    # Adjust Panel 4 y-limits so evidence shows: flat baseline -> climb -> flat plateau
    try:
        ax_ev = axes.flatten()[3]
        valid_logz = res.logz[np.isfinite(res.logz)]
        if len(valid_logz) > 0:
            final_logz = valid_logz[-1]
            final_err = res.logzerr[-1] if (hasattr(res, 'logzerr') and len(res.logzerr) > 0 and np.isfinite(res.logzerr[-1])) else 0.5
            ymax = final_logz + max(3.0 * final_err, 0.5)

            # Determine stable prior baseline, skipping initial 1% transient points
            n_pts = len(valid_logz)
            i0 = min(max(int(0.01 * n_pts), 5), n_pts // 4)
            stable_min = np.min(valid_logz[i0:])
            total_drop = final_logz - stable_min

            if total_drop > 0.5:
                ymin = stable_min - 0.05 * total_drop
            else:
                ymin = final_logz - 5.0

            ax_ev.set_ylim(ymin, ymax)
    except Exception as e:
        logging.warning(f"Could not adjust runplot evidence y-limits: {e}")

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
    ms = './casa_dir_freq/M95_C5+C2_cont93_uvtable.ms'
    msx = './casa_dir_freq/M95_C5+C2_cont93_trimmedXX.ms'
    msy = './casa_dir_freq/M95_C5+C2_cont93_trimmedYY.ms'

    data_imgname = './casa_dir_freq/M95_cont93GHz_auto'
    resid_imgname = './casa_dir_freq/M95_cont93GHz_residual'

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
    # 3b. Visibility-Domain uvplot Diagnostic
    # --------------------------------------------------------------------------
    chi2_vis_x = np.sum(w_datx * (np.real(resid_visx)**2 + np.imag(resid_visx)**2))
    dof_x = 2 * len(vis_datx) - ndim
    red_chi2_x = chi2_vis_x / dof_x
    logging.info(f"Visibility-domain XX fit: chi2 = {chi2_vis_x:.2e}, dof = {dof_x}, reduced chi2 = {red_chi2_x:.4f}")

    logging.info('Generating visibility radial profile via uvplot...')
    try:
        uv_data = UVTable(
            uvtable=[u_datx, v_datx, Re_datx, Im_datx, w_datx],
            columns=['u', 'v', 'Re', 'Im', 'weights']
        )
        uv_mod = UVTable(
            uvtable=[u_datx, v_datx, np.real(model), np.imag(model), w_datx],
            columns=['u', 'v', 'Re', 'Im', 'weights']
        )
        uvdist_max = np.max(np.hypot(u_datx, v_datx))
        uvbin_size = uvdist_max / 40.0

        axes_uv = uv_data.plot(color='black', linestyle='.', label='Observed Data (XX)', uvbin_size=uvbin_size)
        uv_mod.plot(color='crimson', linestyle='-', label='Model Best Fit', axes=list(axes_uv), uvbin_size=uvbin_size, yerr=False)

        fig_uv = axes_uv[0].figure
        axes_uv[0].set_title(f'UVPlot Visibility Radial Profile: {fittype} (reduced $\\chi^2 = {red_chi2_x:.2f}$)')
        axes_uv[0].legend(loc='upper right')
        axes_uv[1].legend(loc='upper right')
        fig_uv.tight_layout()
        pp.savefig(fig_uv)
        plt.close(fig_uv)
        logging.info('Successfully saved uvplot visibility diagnostics plot...')
    except Exception as e:
        logging.warning(f"Could not generate uvplot diagnostic: {e}")

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
    ring_model_intrinsic_mjy = ring_model_intrinsic / 1e6

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
        safe_add_beam(ax, header=hdr_data, frame=False, color='white', corner='bottom left')
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
    pp.savefig(fig)
    plt.close(fig)

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

    axes = [ax1, ax2, ax3]
    for ax in axes:
        ax.set_autoscale_on(False)
        safe_add_beam(ax, header=hdr_data, frame=False, color='white', corner='bottom left')

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
    pp.savefig(fig)
    plt.close(fig)

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

    axes = [ax1, ax2, ax3]
    for ax in axes:
        safe_add_beam(ax, header=hdr_data, frame=False, color='white', corner='bottom left')

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
    pp.savefig(fig)
    plt.close(fig)

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
        cutout_mod_int = Cutout2D(ring_model_intrinsic_mjy, blob_coord, blob_box_size, wcs=mod_wcs)
        cutout_res = Cutout2D(resid_plot, blob_coord, blob_box_size, wcs=data_wcs)

        cutouts = [
            ('Observed CLEAN Image', cutout_data),
            ('Model Sky Intensity (Beam Convolved)', cutout_mod),
            ('Intrinsic Sky Model (Unconvolved)', cutout_mod_int),
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

            ny_cut, nx_cut = cut_data.shape

            # 1D RA brightness profile (horizontal slice across all X at Y=cy)
            ra_profile = np.mean(cut_data[cy - 1 : cy + 2, :], axis=0)
            x_indices = np.arange(nx_cut)
            pixel_world_ra = cut_wcs.pixel_to_world(x_indices, np.full(nx_cut, cy))
            # Relative RA offset in arcseconds centered on Gaussian center b_ra
            ra_offsets_arcsec = (pixel_world_ra.ra.deg - b_ra) * np.cos(np.radians(b_dec)) * 3600.0

            # 1D Dec brightness profile (vertical slice across all Y at X=cx)
            dec_profile = np.mean(cut_data[:, cx - 1 : cx + 2], axis=1)
            y_indices = np.arange(ny_cut)
            pixel_world_dec = cut_wcs.pixel_to_world(np.full(ny_cut, cx), y_indices)
            # Relative Dec offset in arcseconds centered on Gaussian center b_dec
            dec_offsets_arcsec = (pixel_world_dec.dec.deg - b_dec) * 3600.0

            # --- Layout: Left panel = 2D Zoomed Cutout; Right panels = 1D RA & Dec Profiles ---
            fig = plt.figure(figsize=(24, 8))

            ax_img = plt.subplot(131, projection=cut_wcs)
            im_zoom = ax_img.imshow(cut_data, vmin=vmin_zoom, vmax=vmax_zoom, origin='lower', cmap='inferno', rasterized=True)
            ax_img.text(5, 5, f'{target_name} (Zoom)', color='w', fontsize=16)
            cbar_zoom = plt.colorbar(mappable=im_zoom, ax=ax_img, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)
            cbar_zoom.set_label(r'Specific Intensity (MJy/sr)', size=16)

            # Restoring beam patch
            safe_add_beam(ax_img, header=hdr_data, frame=False, color='white', corner='bottom left')

            # Central blob coordinate marker
            ax_img.scatter(b_ra, b_dec, transform=ax_img.get_transform('world'), color='cyan', marker='+', s=150, linewidth=2)

            # Slice indicators
            ax_img.axhline(cy - 1.5, color='crimson', linestyle=':', linewidth=1.5)
            ax_img.axhline(cy + 1.5, color='crimson', linestyle=':', linewidth=1.5)
            ax_img.axvline(cx - 1.5, color='dodgerblue', linestyle=':', linewidth=1.5)
            ax_img.axvline(cx + 1.5, color='dodgerblue', linestyle=':', linewidth=1.5)

            ax_img.set_xlabel(r"Right Ascension (J2000)", size=20)
            ax_img.set_ylabel(r"Declination (J2000)", size=20, labelpad=1)

            # Panel 2: 1D RA Profile
            ax_ra = plt.subplot(132)
            ax_ra.plot(ra_offsets_arcsec, ra_profile, color='crimson', lw=2.5)
            ax_ra.axvline(0.0, color='gray', linestyle='--', alpha=0.7)
            ax_ra.set_xlim(ra_offsets_arcsec[0], ra_offsets_arcsec[-1])
            ax_ra.set_ylim(ylim_min, ylim_max)
            ax_ra.set_xlabel(r"$\Delta$RA Offset (arcsec)", size=20)
            ax_ra.set_ylabel(r"Specific Intensity (MJy/sr)", size=20)
            ax_ra.set_title(r"1D RA Profile ($\pm 1$ Dec pixel avg)", size=18)
            ax_ra.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax_ra.grid(True, alpha=0.3, linestyle=':')

            # Panel 3: 1D Dec Profile
            ax_dec = plt.subplot(133)
            ax_dec.plot(dec_offsets_arcsec, dec_profile, color='dodgerblue', lw=2.5)
            ax_dec.axvline(0.0, color='gray', linestyle='--', alpha=0.7)
            ax_dec.set_xlim(dec_offsets_arcsec[0], dec_offsets_arcsec[-1])
            ax_dec.set_ylim(ylim_min, ylim_max)
            ax_dec.set_xlabel(r"$\Delta$Dec Offset (arcsec)", size=20)
            ax_dec.set_ylabel(r"Specific Intensity (MJy/sr)", size=20)
            ax_dec.set_title(r"1D Dec Profile ($\pm 1$ RA pixel avg)", size=18)
            ax_dec.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax_dec.grid(True, alpha=0.3, linestyle=':')

            blob_label = f"Peak {b_idx + 1}" if "2peak" in fittype else (f"Blob {b_idx + 1}" if len(dynest_coords) > 1 else "Blob")
            fig.suptitle(f"{target_name} - {blob_label} Zoom ($2'' \\times 2''$)", size=26)
            fig.subplots_adjust(hspace=0.2, wspace=0.28, bottom=0.15)
            pp.savefig(fig)
            plt.close(fig)

    # --------------------------------------------------------------------------
    # 8. Summary Table of Posterior Estimates and Derived Quantities
    # --------------------------------------------------------------------------
    logging.info('Rendering summary table page to PDF...')
    render_summary_table_page(res, pars_bf, weights, fittype, pp)

    pp.close()
    logging.info('Post-processing and figure generation finished successfully!')

if __name__ == '__main__':
    main()
