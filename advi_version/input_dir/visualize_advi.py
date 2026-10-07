r"""
ADVI Visualization, Diagnostics, Residual Imaging, and Post-Processing
========================================================================
This script post-processes the results of an Automatic Differentiation Variational
Inference (ADVI) run:
1. Restores variational optimization state from `./output/<fittype>_advi_checkpoint.npz`.
2. Extracts variational parameters, posterior draws, MAP joint sample, and marginal posterior medians.
3. Generates ADVI diagnostic figures:
   - ELBO convergence progression across optimization iterations.
   - Posterior corner plot showing 1D marginals and 2D joint contours.
4. Evaluates best-fit synthetic visibilities (posterior medians) via Galario and subtracts them from
   the observed data to produce residual visibilities: $V_{\text{resid}} = V_{\text{obs}} - V_{\text{mod}}$.
5. Implants residual visibilities into CASA Measurement Sets (`.ms`) and executes
   synthesis dirty imaging (`tclean` with `niter=0`).
6. Convolves the intrinsic sky model with the synthesised beam for like-for-like comparison
   against the restored CLEAN data image.
7. Generates multi-panel comparison figures showing:
   [Observed CLEAN Image | Model Sky Intensity (Beam Convolved) | Residual Visibilities]
   with and without contour overlays, reference cluster positions, 1D horizontal &
   vertical profile cuts, and a publication-grade summary table.
8. Saves all figures into a multipage PDF: `./output/<fittype>_advi_plots.pdf`.
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
try:
    from astropy.visualization.wcsaxes import add_beam
except ImportError:
    try:
        from astropy.visualization.wcsaxes.patches import add_beam
    except ImportError:
        add_beam = None
from matplotlib import rc
from galario.double import get_image_size
from casatools import table
from casatasks import exportfits, imsubimage, tclean, split, impbcor, imsmooth

import argparse
from model_prof import model_prof, model_addon
from radec_calc import sun_radec, dynest_radec, make_model_wcs
import prior_advi
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
    Omega_beam = pi / (4 * ln(2)) * bmaj * bmin
    """
    hdr = fits.open(infile)[0].header
    bmaj = hdr['BMAJ']
    bmin = hdr['BMIN']
    omega_bm_deg2 = (np.pi / (4.0 * np.log(2.0))) * bmaj * bmin
    omega_bm_sr = omega_bm_deg2 / (180.0 / np.pi)**2
    return omega_bm_sr


def img_prepper(fitsimg):
    r"""
    Loads a FITS image, extracts WCS coordinates, converts units, and crops a 2D cutout.
    """
    center = SkyCoord('10h43m57.7330s', '+11d42m12.9996s', frame='icrs')
    box_bkg = [36 * u.arcsecond, 36 * u.arcsecond]

    im = fits.open(fitsimg)[0]
    im_wcs = WCS(im.header, naxis=2)
    logging.info(im_wcs)
    conv = jybm_to_jysr(infile=fitsimg)
    im_plot = Cutout2D(im.data / conv, center, box_bkg, wcs=im_wcs)
    return im_plot.wcs, im_plot.data


def convolve_model_with_beam(model_2d, bmaj_deg, bmin_deg, bpa_deg, pixarcsec):
    r"""
    Convolves an intrinsic 2D sky brightness model with the synthesis restoring beam.
    """
    if bmaj_deg <= 0 or bmin_deg <= 0:
        return model_2d

    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_maj_pix = (bmaj_deg * 3600.0 * fwhm_to_sigma) / pixarcsec
    sigma_min_pix = (bmin_deg * 3600.0 * fwhm_to_sigma) / pixarcsec

    theta_rad = np.radians(bpa_deg)
    beam_kernel = Gaussian2DKernel(
        x_stddev=sigma_min_pix,
        y_stddev=sigma_maj_pix,
        theta=theta_rad
    )
    convolved = convolve_fft(model_2d, beam_kernel, normalize_kernel=True)
    return convolved


def safe_add_beam(ax, header=None, frame=False, color='white', corner='bottom left'):
    r"""
    Safely adds a synthesised restoring beam patch to a celestial WCSAxes subplot.
    """
    if header is None or 'BMAJ' not in header:
        return
    if add_beam is not None:
        try:
            add_beam(ax, header=header, frame=frame, color=color, corner=corner)
            return
        except Exception as e:
            logging.warning(f"Could not add beam to axis: {e}")
    try:
        from matplotlib.patches import Ellipse
        bmaj_deg = header.get('BMAJ', 0.0)
        bmin_deg = header.get('BMIN', 0.0)
        bpa_deg = header.get('BPA', 0.0)
        cdelt = abs(header.get('CDELT2', header.get('CD2_2', 1.0 / 3600.0)))
        bmaj_pix = bmaj_deg / cdelt
        bmin_pix = bmin_deg / cdelt
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x0 = min(xlim) + 0.1 * abs(xlim[1] - xlim[0])
        y0 = min(ylim) + 0.1 * abs(ylim[1] - ylim[0])
        beam_patch = Ellipse((x0, y0), width=bmin_pix, height=bmaj_pix, angle=bpa_deg,
                             edgecolor=color, facecolor=color, alpha=0.8)
        ax.add_patch(beam_patch)
    except Exception as e:
        logging.warning(f"Could not add fallback beam patch: {e}")


def circular_mean_and_dispersion(samples, weights=None, period=180.0):
    r"""
    Computes the circular mean and angular dispersion for periodic parameters.
    """
    samples = np.asarray(samples)
    if weights is None:
        weights = np.ones_like(samples)
    weights = weights / np.sum(weights)

    scale = 2.0 * np.pi / period
    theta = samples * scale
    C = np.sum(weights * np.cos(theta))
    S = np.sum(weights * np.sin(theta))
    mean_angle = np.arctan2(S, C) / scale
    mean_angle = float(mean_angle % period)

    R = np.sqrt(C**2 + S**2)
    R = min(max(R, 1e-12), 1.0)
    circular_std = float(np.sqrt(-2.0 * np.log(R)) / scale)
    return mean_angle, circular_std


def compute_posterior_median_pars(samples, labels, ndim):
    r"""
    Marginal posterior median parameter vector (circular mean for angle parameters).
    """
    pars_median = np.median(samples, axis=0).copy()
    for i in range(ndim):
        label_lower = labels[i].lower()
        if 'posangle' in label_lower or (i == 4):
            pars_median[i], _ = circular_mean_and_dispersion(samples[:, i], period=180.0)
        elif 'ang' in label_lower:
            pars_median[i], _ = circular_mean_and_dispersion(samples[:, i], period=360.0)
    return pars_median


def overlay_corner_marginal_lines(fig, ndim, pars, color, linestyle='-', linewidth=1.8):
    r"""Overlay vertical lines on 1D marginal axes of a corner plot figure."""
    for i in range(ndim):
        ax_idx = i * (i + 3) // 2
        fig.axes[ax_idx].axvline(pars[i], color=color, linestyle=linestyle, linewidth=linewidth)


def plot_elbo_convergence(elbo_history, converged):
    r"""
    Generates a publication-quality diagnostic figure showing the ELBO convergence trajectory.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    iters = np.arange(1, len(elbo_history) + 1)
    elbo = np.array(elbo_history)

    ax.plot(iters, elbo, color='dodgerblue', alpha=0.45, label='Stochastic ELBO')
    window = min(30, max(5, len(elbo) // 20))
    if len(elbo) >= window:
        rolling_mean = np.convolve(elbo, np.ones(window) / window, mode='valid')
        roll_iters = iters[window - 1 :]
        ax.plot(roll_iters, rolling_mean, color='darkblue', lw=2.2, label=f'Rolling Mean (w={window})')

    ax.set_xlabel('Optimization Iteration', fontsize=16, fontweight='bold')
    ax.set_ylabel('Evidence Lower Bound (ELBO)', fontsize=16, fontweight='bold')
    title_suffix = " (Converged via tol_rel_obj)" if converged else " (Completed Max Iterations)"
    ax.set_title(f"ADVI Optimization Trajectory{title_suffix}", fontsize=18, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', fontsize=14, frameon=True)

    fig.tight_layout()
    return fig


def render_summary_table_page(samples, pars_map, fittype, pp):
    r"""
    Constructs and saves the final summary table PDF page containing:
    - Free parameters fitted by ADVI
    - Physical conversions:
        - Ring Flux (Jy) [after Ring LogFlux]
        - Ring LogPeak (log(Jy/sr)) [after Ring Flux]
        - Ring LogSigma (log(arcsec)) [after Ring LogPeak]
        - Ring Sigma (arcsec) [after Ring LogSigma]
        - Ring FWHM (arcsec) [after Ring Sigma]
        - Equivalent Flux, LogPeak, Sigma, and FWHM for modeled blob component.
    - Two-tiered table header:
        - (Prior information) spanning Prior Low & Prior High
        - (Results) spanning MAP Estimate & Median Estimate (+84% / -16%)
    """
    ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
    FWHM_FACTOR = 2.354820045

    def get_quantiles(arr):
        return np.percentile(np.asarray(arr), [15.865, 50.0, 84.135])

    def fmt_val(v):
        if v is None or np.isnan(v):
            return "-"
        abs_v = abs(v)
        if abs_v == 0.0:
            return "0.0"
        if abs_v >= 1e4 or (abs_v < 1e-3 and abs_v > 0):
            return f"{v:.3e}"
        elif abs_v >= 100.0:
            return f"{v:.2f}"
        else:
            return f"{v:.4f}"

    def fmt_quantiles(q_vals):
        median = q_vals[1]
        p1 = max(0.0, q_vals[2] - median)
        m1 = max(0.0, median - q_vals[0])
        return f"{fmt_val(median)} (+{fmt_val(p1)} / -{fmt_val(m1)})"

    table_rows = []

    def add_table_row(name, unit, p_low, p_high, map_val, q_vals):
        param_label = f"{name} ({unit})" if unit else name
        table_rows.append([
            param_label,
            fmt_val(p_low),
            fmt_val(p_high),
            fmt_val(map_val),
            fmt_quantiles(q_vals)
        ])

    # 1. Ring Component
    q_lf = get_quantiles(samples[:, 0])
    add_table_row("Ring LogFlux", "log(Jy)", prior_advi.RING_PRIOR_RANGES[0, 0], prior_advi.RING_PRIOR_RANGES[0, 1],
                  pars_map[0], q_lf)

    flux_samples = 10.0**samples[:, 0]
    q_flux = get_quantiles(flux_samples)
    add_table_row("Ring Flux", "Jy", 10.0**prior_advi.RING_PRIOR_RANGES[0, 0], 10.0**prior_advi.RING_PRIOR_RANGES[0, 1],
                  10.0**pars_map[0], q_flux)

    cos_inc_samples = np.maximum(np.cos(np.radians(samples[:, 3])), 0.05)
    area_ring_samples = ((2.0 * np.pi)**1.5) * (samples[:, 2] * ARCSEC_TO_RAD) * (10.0**samples[:, 1] * ARCSEC_TO_RAD) * cos_inc_samples
    ring_peak_samples = flux_samples / np.maximum(area_ring_samples, 1e-30)
    ring_logpeak_samples = np.log10(np.maximum(ring_peak_samples, 1e-30))
    q_rpeak = get_quantiles(ring_logpeak_samples)

    cos_inc_map = max(float(np.cos(np.radians(pars_map[3]))), 0.05)
    area_ring_map = ((2.0 * np.pi)**1.5) * (pars_map[2] * ARCSEC_TO_RAD) * (10.0**pars_map[1] * ARCSEC_TO_RAD) * cos_inc_map
    ring_peak_map = (10.0**pars_map[0]) / max(area_ring_map, 1e-30)
    ring_logpeak_map = np.log10(max(ring_peak_map, 1e-30))

    add_table_row("Ring LogPeak", "log(Jy/sr)", prior_advi.COMMON_RING_PRIORS[0, 0], prior_advi.COMMON_RING_PRIORS[0, 1],
                  ring_logpeak_map, q_rpeak)

    q_ls = get_quantiles(samples[:, 1])
    add_table_row("Ring LogSigma", "log(arcsec)", prior_advi.RING_PRIOR_RANGES[1, 0], prior_advi.RING_PRIOR_RANGES[1, 1],
                  pars_map[1], q_ls)

    sigma_samples = 10.0**samples[:, 1]
    q_sigma = get_quantiles(sigma_samples)
    add_table_row("Ring Sigma", "arcsec", prior_advi.COMMON_RING_PRIORS[1, 0], prior_advi.COMMON_RING_PRIORS[1, 1],
                  10.0**pars_map[1], q_sigma)

    fwhm_samples = FWHM_FACTOR * sigma_samples
    q_fwhm = get_quantiles(fwhm_samples)
    add_table_row("Ring FWHM", "arcsec", FWHM_FACTOR * prior_advi.COMMON_RING_PRIORS[1, 0], FWHM_FACTOR * prior_advi.COMMON_RING_PRIORS[1, 1],
                  FWHM_FACTOR * 10.0**pars_map[1], q_fwhm)

    ring_param_names = ["Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec"]
    ring_param_units = ["arcsec", "degrees", "degrees", "arcsec", "arcsec"]
    for p_idx, (p_name, p_unit) in enumerate(zip(ring_param_names, ring_param_units), start=2):
        q_p = get_quantiles(samples[:, p_idx])
        add_table_row(p_name, p_unit, prior_advi.COMMON_RING_PRIORS[p_idx, 0], prior_advi.COMMON_RING_PRIORS[p_idx, 1],
                      pars_map[p_idx], q_p)

    def add_blob_component(blob_label, idx_flux, idx_sigma, prior_logflux_range, prior_peak_log_range, prior_sigma_range):
        # 1. LogFlux
        q_lf = get_quantiles(samples[:, idx_flux])
        add_table_row(f"{blob_label} LogFlux", "log(Jy)", prior_logflux_range[0], prior_logflux_range[1],
                      pars_map[idx_flux], q_lf)

        # 2. Flux (Jy)
        b_flux_samples = 10.0**samples[:, idx_flux]
        q_bflux = get_quantiles(b_flux_samples)
        add_table_row(f"{blob_label} Flux", "Jy", 10.0**prior_logflux_range[0], 10.0**prior_logflux_range[1],
                      10.0**pars_map[idx_flux], q_bflux)

        # 3. LogPeak (log(Jy/sr))
        b_area_samples = 2.0 * np.pi * ((10.0**samples[:, idx_sigma] * ARCSEC_TO_RAD)**2)
        b_peak_samples = b_flux_samples / np.maximum(b_area_samples, 1e-30)
        b_logpeak_samples = np.log10(np.maximum(b_peak_samples, 1e-30))
        q_bpeak = get_quantiles(b_logpeak_samples)

        b_area_map = 2.0 * np.pi * ((10.0**pars_map[idx_sigma] * ARCSEC_TO_RAD)**2)
        b_peak_map = (10.0**pars_map[idx_flux]) / max(b_area_map, 1e-30)
        b_logpeak_map = np.log10(max(b_peak_map, 1e-30))
        add_table_row(f"{blob_label} LogPeak", "log(Jy/sr)", prior_peak_log_range[0], prior_peak_log_range[1],
                      b_logpeak_map, q_bpeak)

        # 4. LogSigma
        prior_logsigma_min = np.log10(prior_sigma_range[0])
        prior_logsigma_max = np.log10(prior_sigma_range[1])
        q_ls = get_quantiles(samples[:, idx_sigma])
        add_table_row(f"{blob_label} LogSigma", "log(arcsec)", prior_logsigma_min, prior_logsigma_max,
                      pars_map[idx_sigma], q_ls)

        # 5. Sigma (arcsec)
        b_sigma_samples = 10.0**samples[:, idx_sigma]
        q_bsigma = get_quantiles(b_sigma_samples)
        add_table_row(f"{blob_label} Sigma", "arcsec", prior_sigma_range[0], prior_sigma_range[1],
                      10.0**pars_map[idx_sigma], q_bsigma)

        # 6. FWHM (arcsec)
        b_fwhm_samples = FWHM_FACTOR * b_sigma_samples
        q_bfwhm = get_quantiles(b_fwhm_samples)
        add_table_row(f"{blob_label} FWHM", "arcsec", FWHM_FACTOR * prior_sigma_range[0], FWHM_FACTOR * prior_sigma_range[1],
                      FWHM_FACTOR * 10.0**pars_map[idx_sigma], q_bfwhm)

    # 2. Model-Specific Blob Components
    if fittype == 'twod_gauss1blob':
        add_blob_component("B1", 7, 8,
                           prior_advi.GAUSS1BLOB_PRIOR_RANGES[7],
                           prior_advi.GAUSS1BLOB_USER_PRIORS[0],
                           prior_advi.GAUSS1BLOB_USER_PRIORS[1])
        q_dist = get_quantiles(samples[:, 9])
        add_table_row("B1 Dist", "arcsec", prior_advi.GAUSS1BLOB_USER_PRIORS[2, 0], prior_advi.GAUSS1BLOB_USER_PRIORS[2, 1],
                      pars_map[9], q_dist)
        q_ang = get_quantiles(samples[:, 10])
        add_table_row("B1 Angle", "degrees", prior_advi.GAUSS1BLOB_USER_PRIORS[3, 0], prior_advi.GAUSS1BLOB_USER_PRIORS[3, 1],
                      pars_map[10], q_ang)

    elif fittype == 'twod_gauss2blob':
        for k, name in [(1, 'B1'), (2, 'B2')]:
            idx_base = 7 + (k - 1) * 4
            u_base = (k - 1) * 4
            add_blob_component(name, idx_base, idx_base + 1,
                               prior_advi.GAUSS2BLOB_PRIOR_RANGES[idx_base],
                               prior_advi.GAUSS2BLOB_USER_PRIORS[u_base],
                               prior_advi.GAUSS2BLOB_USER_PRIORS[u_base + 1])
            q_dist = get_quantiles(samples[:, idx_base + 2])
            add_table_row(f"{name} Dist", "arcsec", prior_advi.GAUSS2BLOB_USER_PRIORS[u_base + 2, 0], prior_advi.GAUSS2BLOB_USER_PRIORS[u_base + 2, 1],
                          pars_map[idx_base + 2], q_dist)
            q_ang = get_quantiles(samples[:, idx_base + 3])
            add_table_row(f"{name} Angle", "degrees", prior_advi.GAUSS2BLOB_USER_PRIORS[u_base + 3, 0], prior_advi.GAUSS2BLOB_USER_PRIORS[u_base + 3, 1],
                          pars_map[idx_base + 3], q_ang)

    elif fittype == 'twod_gauss3blob':
        for k, name in [(1, 'B1'), (2, 'B2'), (3, 'B3')]:
            idx_base = 7 + (k - 1) * 4
            u_base = (k - 1) * 4
            add_blob_component(name, idx_base, idx_base + 1,
                               prior_advi.GAUSS3BLOB_PRIOR_RANGES[idx_base],
                               prior_advi.GAUSS3BLOB_USER_PRIORS[u_base],
                               prior_advi.GAUSS3BLOB_USER_PRIORS[u_base + 1])
            q_dist = get_quantiles(samples[:, idx_base + 2])
            add_table_row(f"{name} Dist", "arcsec", prior_advi.GAUSS3BLOB_USER_PRIORS[u_base + 2, 0], prior_advi.GAUSS3BLOB_USER_PRIORS[u_base + 2, 1],
                          pars_map[idx_base + 2], q_dist)
            q_ang = get_quantiles(samples[:, idx_base + 3])
            add_table_row(f"{name} Angle", "degrees", prior_advi.GAUSS3BLOB_USER_PRIORS[u_base + 3, 0], prior_advi.GAUSS3BLOB_USER_PRIORS[u_base + 3, 1],
                          pars_map[idx_base + 3], q_ang)

    col_labels = ["Parameter", "Low", "High", "MAP Estimate", "Median Estimate"]
    col_widths = [0.32, 0.15, 0.15, 0.16, 0.22]
    top_widths = [0.32, 0.30, 0.38]

    fig_height = max(8.5, 0.38 * len(table_rows) + 2.0)
    fig_tab, ax_tab = plt.subplots(figsize=(14, fig_height))
    ax_tab.axis('off')

    n_rows = len(table_rows) + 1
    h_row = 0.85 / (n_rows + 1)
    main_h = n_rows * h_row
    top_h = h_row
    bottom_main = 0.05
    left = 0.05
    width = 0.90

    tab_top = ax_tab.table(
        cellText=[['', 'Prior - U(Low, High)', 'Results']],
        colWidths=top_widths,
        bbox=[left, bottom_main + main_h, width, top_h],
        cellLoc='center'
    )

    tab_main = ax_tab.table(
        cellText=table_rows,
        colLabels=col_labels,
        colWidths=col_widths,
        bbox=[left, bottom_main, width, main_h],
        cellLoc='center'
    )

    tab_top[(0, 0)].set_visible(False)
    tab_top[(0, 1)].set_facecolor('#c0c0c0')
    tab_top[(0, 1)].set_text_props(weight='bold', size=11)
    tab_top[(0, 2)].set_facecolor('#c0c0c0')
    tab_top[(0, 2)].set_text_props(weight='bold', size=11)

    for c in range(5):
        tab_main[(0, c)].set_facecolor('#d9d9d9')
        tab_main[(0, c)].set_text_props(weight='bold', size=10)

    for row_idx in range(1, len(table_rows) + 1):
        bg_color = '#f5f5f5' if row_idx % 2 == 0 else '#ffffff'
        for col_idx in range(5):
            tab_main[(row_idx, col_idx)].set_facecolor(bg_color)
            tab_main[(row_idx, col_idx)].set_text_props(size=10)

    clean_fittype = fittype.replace('_', r'\_')
    fig_tab.suptitle(f"ADVI Posterior Summary and Derived Parameters: {clean_fittype}", fontsize=16, y=0.98)
    pp.savefig(fig_tab, bbox_inches='tight')
    plt.close(fig_tab)
    logging.info('Successfully saved final summary table page to PDF...')


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and post-process ADVI fitting results."
    )
    parser.add_argument("fittype", type=str, choices=['twod_gaussring', 'twod_gauss1blob', 'twod_gauss2blob', 'twod_gauss3blob'],
                        help="Model configuration identifier ('twod_gaussring', 'twod_gauss1blob', 'twod_gauss2blob', 'twod_gauss3blob').")
    pargs = parser.parse_args()

    fittype = pargs.fittype

    os.makedirs('./output', exist_ok=True)
    logfile = f'./output/{fittype}_advi_vis.log'
    logging.basicConfig(
        filename=logfile,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )
    logging.info("==========ADVI Visualization Output==========")
    logging.info(f'Fittype: {fittype}')

    # --------------------------------------------------------------------------
    # 1. Restore ADVI Checkpoint & Posterior Parameter Summaries
    # --------------------------------------------------------------------------
    check_file = f'./output/{fittype}_advi_checkpoint.npz'
    if not os.path.exists(check_file):
        msg = f"Checkpoint file '{check_file}' not found. Please run 'run_advi.py {fittype}' first."
        logging.error(msg)
        raise FileNotFoundError(msg)

    data = np.load(check_file, allow_pickle=True)
    samples = data['samples']
    pars_map = data['best_fit'].copy()
    elbo_history = data['elbo_history']
    converged = bool(data['converged'])
    labels = list(data['labels'])
    units = list(data['units'])
    ndim = int(data['ndim'])
    pars_bf = compute_posterior_median_pars(samples, labels, ndim)

    logging.info(f"Read in results from saved checkpoint file {check_file}")
    logging.info(f"Dimensions for {fittype}: {ndim}")
    logging.info(f"Parameters for {fittype}: {labels}")
    logging.info(f"Coherent Joint MAP Parameters: {pars_map}")
    logging.info(f"Marginal Posterior Median Parameters (used for best-fit model): {pars_bf}")

    # Log marginal quantiles and circular stats
    logging.info("Marginal posterior medians and 68% credible intervals:")
    for i in range(ndim):
        label_lower = labels[i].lower()
        if 'posangle' in label_lower or (i == 4):
            c_mean, c_std = circular_mean_and_dispersion(samples[:, i], period=180.0)
            logging.info(f"  {labels[i]:<25} : Circular Axial Mean = {c_mean:.4f} deg (circ std = {c_std:.4f} deg) [180-deg axial]")
        elif 'ang' in label_lower:
            c_mean, c_std = circular_mean_and_dispersion(samples[:, i], period=360.0)
            logging.info(f"  {labels[i]:<25} : Circular Mean = {c_mean:.4f} deg (circ std = {c_std:.4f} deg) [360-deg circular]")

        q16, q50, q84 = np.percentile(samples[:, i], [15.865, 50.0, 84.135])
        m1 = q50 - q16
        p1 = q84 - q50
        if 'flux' in label_lower:
            flux_mjy = (10.0**q50) * 1e3
            logging.info(f"  {labels[i]:<25} : {q50:.4f} (+{p1:.4f} / -{m1:.4f}) {units[i]}  -->  Flux = {flux_mjy:.4f} mJy")
        elif 'sigma' in label_lower:
            sigma_arcsec = 10.0**q50
            fwhm_arcsec = 2.35482 * sigma_arcsec
            logging.info(f"  {labels[i]:<25} : {q50:.4f} (+{p1:.4f} / -{m1:.4f}) {units[i]}  -->  Sigma = {sigma_arcsec:.4f}\", FWHM = {fwhm_arcsec:.4f}\"")
        else:
            logging.info(f"  {labels[i]:<25} : {q50:.4f} (+{p1:.4f} / -{m1:.4f}) {units[i]}")

    pdf_plot = f'./output/{fittype}_plots.pdf'
    pp = PdfPages(pdf_plot)
    logging.info(f'Plot File: {pdf_plot}')

    # --------------------------------------------------------------------------
    # 2. ADVI ELBO Convergence & Corner Plots
    # --------------------------------------------------------------------------
    fig_elbo = plot_elbo_convergence(elbo_history, converged)
    pp.savefig(fig_elbo)
    plt.close(fig_elbo)
    logging.info('Successfully saved the ELBO convergence summary plot...')

    # Corner plot. A 19-parameter figure saved from every posterior draw at
    # 300 dpi exhausts the container during the 3-blob model. Contours are
    # drawn from a fixed 1000-draw subset; the results table and best-fit
    # model still use the full posterior.
    disp_labels = [f"{l}\n[{u}]" for l, u in zip(labels, units)]
    rng = np.random.default_rng(42)
    n_corner = min(1000, len(samples))
    corner_samples = samples[rng.choice(len(samples), n_corner, replace=False)]
    fig_corner = corner.corner(
        corner_samples,
        labels=disp_labels,
        quantiles=[0.15865, 0.5, 0.84135],
        show_titles=True,
        title_fmt='.3f',
        color='dodgerblue',
        truths=pars_bf,
        truth_color='forestgreen'
    )
    #overlay_corner_marginal_lines(fig_corner, ndim, pars_map, color='crimson', linestyle='-', linewidth=1.8)
    from matplotlib.lines import Line2D
    fig_corner.legend(
        handles=[
            Line2D([0], [0], color='forestgreen', lw=2, label='Posterior median (best fit)'),
            #Line2D([0], [0], color='crimson', lw=2, label='Joint MAP'),
        ],
        loc='upper right',
        fontsize=10,
        frameon=True,
    )
    pp.savefig(fig_corner, dpi=100)
    plt.close(fig_corner)
    logging.info('Successfully saved the corner plot...')

    # --------------------------------------------------------------------------
    # 3. Load Observed Visibilities for Residual Calculation
    # --------------------------------------------------------------------------
    ms = './casa_dir_freq/M95_C5+C2_cont93_uvtable.ms'
    msx = './casa_dir_freq/M95_C5+C2_cont93_trimmedXX.ms'
    msy = './casa_dir_freq/M95_C5+C2_cont93_trimmedYY.ms'

    data_imgname = './casa_dir_freq/M95_cont93GHz_auto'
    resid_imgname = './casa_dir_freq/M95_cont93GHz_residual'

    raw_x = np.require(np.loadtxt(msx + '.uvtable.txt', unpack=True), requirements='C')
    if len(raw_x) >= 6:
        u_datx, v_datx, Re_datx, Im_datx, w_datx, freq_x = raw_x[:6]
        wavelength_x = 299792458.0 / freq_x
        u_datx = u_datx / wavelength_x
        v_datx = v_datx / wavelength_x
    else:
        u_datx, v_datx, Re_datx, Im_datx, w_datx = raw_x[:5]
    vis_datx = np.array(Re_datx + 1j * Im_datx, dtype=np.complex256)

    raw_y = np.require(np.loadtxt(msy + '.uvtable.txt', unpack=True), requirements='C')
    if len(raw_y) >= 6:
        u_daty, v_daty, Re_daty, Im_daty, w_daty, freq_y = raw_y[:6]
        wavelength_y = 299792458.0 / freq_y
        u_daty = u_daty / wavelength_y
        v_daty = v_daty / wavelength_y
    else:
        u_daty, v_daty, Re_daty, Im_daty, w_daty = raw_y[:5]
    vis_daty = np.array(Re_daty + 1j * Im_daty, dtype=np.complex256)

    nxy, dxy = get_image_size(u_datx, v_datx, verbose=True)
    dxy_arcsec = dxy * 206265.0
    args_vis = (nxy, dxy)
    vis_x_dat = (u_datx, v_datx, Re_datx, Im_datx, w_datx)

    logging.info('Evaluating best-fit model visibilities via Galario...')
    model = model_prof(pars_bf, args_vis, vis_x_dat, 'vis', fittype)
    logging.info('Successfully computed best fit model visibilities...')

    resid_visx = vis_datx - model
    resid_visy = vis_daty - model
    logging.info('Residual visibilities calculated...')

    # 3b. Visibility-Domain uvplot Diagnostic
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

        fig_uv = plt.figure(figsize=(24, 10))
        gs = fig_uv.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.0)
        ax_uv1 = fig_uv.add_subplot(gs[0])
        ax_uv2 = fig_uv.add_subplot(gs[1], sharex=ax_uv1)
        axes_list = [ax_uv1, ax_uv2]

        uv_data.plot(axes=axes_list, color='black', linestyle='.', label='Observed Data', uvbin_size=uvbin_size)
        uv_mod.plot(axes=axes_list, color='crimson', linestyle='-', label='Model Best Fit', uvbin_size=uvbin_size, yerr=False)

        ax_uv1.yaxis.set_label_coords(-0.05, 0.5)
        ax_uv2.yaxis.set_label_coords(-0.05, 0.5)
        ax_uv1.legend(loc='upper right', fontsize=16)
        if ax_uv2.get_legend():
            ax_uv2.get_legend().remove()
        fig_uv.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.12, hspace=0.0)
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

    os.system('rm -rf ' + msx + '.residual.ms')
    os.system('cp -R ' + msx + ' ' + msx + '.residual.ms')
    tb.open(msx + '.residual.ms', nomodify=False)
    tb.putcol('DATA', implant_visx)
    tb.flush()
    tb.close()

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

    os.system('rm -rf ' + data_imgname + '.image.pbcor')
    os.system('rm -rf ' + data_imgname + '.fits')
    os.system('rm -rf ' + data_imgname + '.image.fits')
    os.system('rm -rf ' + resid_imgname + '.image.pbcor')
    os.system('rm -rf ' + resid_imgname + '.fits')
    os.system('rm -rf ' + resid_imgname + '.dirty.fits')

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

    ring_model = convolve_model_with_beam(ring_model_intrinsic, bmaj_deg, bmin_deg, bpa_deg, pixarcsec)
    logging.info('Convolved intrinsic model with restoring beam...')

    phase_center = SkyCoord('10h43m57.7330s', '+11d42m12.9996s', frame='icrs')
    mod_wcs = make_model_wcs(phase_center.ra.deg, phase_center.dec.deg, pixarcsec, shape=(numpix, numpix))

    # Pre-calculate celestial coordinates for the ring peak profile (white dashed ellipse)
    ring_rad = pars_bf[2]
    inc = np.radians(pars_bf[3])
    pa = np.radians(pars_bf[4])
    dRA = pars_bf[5]
    dDec = pars_bf[6]

    theta_ellipse = np.linspace(0, 2 * np.pi, 500)
    x_pa = ring_rad * np.cos(inc) * np.sin(theta_ellipse)
    y_pa = ring_rad * np.cos(theta_ellipse)

    x_shifted = x_pa * np.cos(pa) + y_pa * np.sin(pa)
    y_shifted = -x_pa * np.sin(pa) + y_pa * np.cos(pa)

    ngc3351_center = phase_center.spherical_offsets_by(dRA * u.arcsec, dDec * u.arcsec)
    ellipse_sky = ngc3351_center.spherical_offsets_by(x_shifted * u.arcsec, y_shifted * u.arcsec)
    ellipse_ra = ellipse_sky.ra.deg
    ellipse_dec = ellipse_sky.dec.deg

    # Convert surface brightness from Jy/sr to MJy/sr
    data_plot /= 1e6
    resid_plot /= 1e6
    ring_model /= 1e6
    ring_model_intrinsic_mjy = ring_model_intrinsic / 1e6

    # --- Figure 1: Side-by-Side Comparison (Data, Model, Residuals) ---
    fig = plt.figure(figsize=(24, 8))

    ax1 = plt.subplot(131, projection=data_wcs)
    im1 = ax1.imshow(data_plot, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    cbar1 = plt.colorbar(mappable=im1, ax=ax1, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax2 = plt.subplot(132, projection=mod_wcs)
    im2 = ax2.imshow(ring_model, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    cbar2 = plt.colorbar(mappable=im2, ax=ax2, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax3 = plt.subplot(133, projection=data_wcs)
    im3 = ax3.imshow(resid_plot, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
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
    cbar1 = plt.colorbar(mappable=im1, ax=ax1, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax2 = plt.subplot(132, projection=mod_wcs)
    im2 = ax2.imshow(ring_model, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    cbar2 = plt.colorbar(mappable=im2, ax=ax2, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax3 = plt.subplot(133, projection=data_wcs)
    im3 = ax3.imshow(resid_plot, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    cbar3 = plt.colorbar(mappable=im3, ax=ax3, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    axes = [ax1, ax2, ax3]
    for ax in axes:
        ax.set_autoscale_on(False)
        safe_add_beam(ax, header=hdr_data, frame=False, color='white', corner='bottom left')

    peak_flux = np.percentile(data_plot, 99.95)

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
    cbar1 = plt.colorbar(mappable=im1, ax=ax1, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax2 = plt.subplot(132, projection=mod_wcs)
    im2 = ax2.imshow(ring_model, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    cbar2 = plt.colorbar(mappable=im2, ax=ax2, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    ax3 = plt.subplot(133, projection=data_wcs)
    im3 = ax3.imshow(resid_plot, vmin=np.percentile(data_plot, 1), vmax=np.percentile(data_plot, 99.95), origin='lower', cmap='inferno', rasterized=True)
    cbar3 = plt.colorbar(mappable=im3, ax=ax3, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)

    axes = [ax1, ax2, ax3]
    for ax in axes:
        safe_add_beam(ax, header=hdr_data, frame=False, color='white', corner='bottom left')

    ymc_ids = [15, 6, 18]
    sun_coords = sun_radec(ymc_ids)
    advi_coords = dynest_radec(pars_bf, fittype)

    sun_ra, sun_dec = zip(*sun_coords)
    advi_ra, advi_dec = zip(*advi_coords)

    for ax in axes:
        ax.set_autoscale_on(False)
        ax.plot(ellipse_ra, ellipse_dec, transform=ax.get_transform('world'),
                color='white', linestyle='--', linewidth=1.5, zorder=12, label='Ring Ridge Peak')
        ax.scatter(advi_ra, advi_dec, transform=ax.get_transform('world'), color='blue', marker='x', s=120, linewidth=2, zorder=15, label='Fitted Model')
        ax.scatter(phase_center.ra.deg, phase_center.dec.deg, transform=ax.get_transform('world'), color='yellow', marker='+', s=120, linewidth=2, zorder=15, label='Phase Center')

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

    # Save FITS products
    header_data = data_wcs.to_header()
    header_model = mod_wcs.to_header()

    hdu_data = fits.PrimaryHDU(data=data_plot, header=header_data)
    hdu_model = fits.PrimaryHDU(data=ring_model, header=header_model)
    hdu_resid = fits.PrimaryHDU(data=resid_plot, header=header_data)

    hdu_data.writeto("./output/data_plot.fits", overwrite=True)
    hdu_model.writeto("./output/ring_model.fits", overwrite=True)
    hdu_resid.writeto("./output/resid_plot.fits", overwrite=True)

    # --------------------------------------------------------------------------
    # 7. Zoomed-In Blob Analysis & 1D Brightness Profiles
    # --------------------------------------------------------------------------
    blob_box_size = [2 * u.arcsecond, 2 * u.arcsecond]

    for b_idx, (b_ra, b_dec) in enumerate(advi_coords):
        blob_coord = SkyCoord(b_ra * u.deg, b_dec * u.deg, frame='icrs')

        cutout_data = Cutout2D(data_plot, blob_coord, blob_box_size, wcs=data_wcs)
        cutout_mod = Cutout2D(ring_model, blob_coord, blob_box_size, wcs=mod_wcs)
        cutout_res = Cutout2D(resid_plot, blob_coord, blob_box_size, wcs=data_wcs)
        cutout_mod_int = Cutout2D(ring_model_intrinsic_mjy, blob_coord, blob_box_size, wcs=mod_wcs)

        shared_cut_data = [cutout_data.data, cutout_mod.data, cutout_res.data]
        vmin_shared = min(0.0, min(np.percentile(d, 1) for d in shared_cut_data))
        vmax_shared = max(np.max(d) for d in shared_cut_data)
        ylim_min_shared = vmin_shared - 0.02 * (vmax_shared - vmin_shared) if vmin_shared < 0 else 0.0
        ylim_max_shared = vmax_shared * 1.05

        unconv_data = cutout_mod_int.data
        vmin_unconv = min(0.0, float(np.percentile(unconv_data, 1)))
        vmax_unconv = float(np.max(unconv_data))
        ylim_min_unconv = vmin_unconv - 0.02 * (vmax_unconv - vmin_unconv) if vmin_unconv < 0 else 0.0
        ylim_max_unconv = vmax_unconv * 1.05

        cutouts = [
            ('Observed CLEAN Image', cutout_data, False),
            ('Model Sky Intensity (Beam Convolved)', cutout_mod, False),
            ('Dirty Residual Visibilities', cutout_res, False),
            ('Intrinsic Sky Model (Unconvolved)', cutout_mod_int, True)
        ]

        for target_name, cutout, is_unconv in cutouts:
            cut_data = cutout.data
            cut_wcs = cutout.wcs

            if is_unconv:
                vmin_zoom = vmin_unconv
                vmax_zoom = vmax_unconv
                ylim_min = ylim_min_unconv
                ylim_max = ylim_max_unconv
            else:
                vmin_zoom = vmin_shared
                vmax_zoom = vmax_shared
                ylim_min = ylim_min_shared
                ylim_max = ylim_max_shared

            cx_f, cy_f = cut_wcs.world_to_pixel(blob_coord)
            cx = int(round(float(cx_f)))
            cy = int(round(float(cy_f)))

            ny_cut, nx_cut = cut_data.shape

            ra_profile = np.mean(cut_data[cy - 1 : cy + 2, :], axis=0)
            x_indices = np.arange(nx_cut)
            pixel_world_ra = cut_wcs.pixel_to_world(x_indices, np.full(nx_cut, cy))
            ra_offsets_arcsec = (pixel_world_ra.ra.deg - b_ra) * np.cos(np.radians(b_dec)) * 3600.0

            dec_profile = np.mean(cut_data[:, cx - 1 : cx + 2], axis=1)
            y_indices = np.arange(ny_cut)
            pixel_world_dec = cut_wcs.pixel_to_world(np.full(ny_cut, cx), y_indices)
            dec_offsets_arcsec = (pixel_world_dec.dec.deg - b_dec) * 3600.0

            fig = plt.figure(figsize=(24, 8))

            ax_img = plt.subplot(131, projection=cut_wcs)
            im_zoom = ax_img.imshow(cut_data, vmin=vmin_zoom, vmax=vmax_zoom, origin='lower', cmap='inferno', rasterized=True)
            cbar_zoom = plt.colorbar(mappable=im_zoom, ax=ax_img, orientation='vertical', location='right', pad=0.05, shrink=0.8, aspect=15)
            cbar_zoom.set_label(r'Specific Intensity (MJy/sr)', size=16)

            safe_add_beam(ax_img, header=hdr_data, frame=False, color='white', corner='bottom left')

            ax_img.set_autoscale_on(False)
            ax_img.plot(ellipse_ra, ellipse_dec, transform=ax_img.get_transform('world'),
                        color='white', linestyle='--', linewidth=1.5, zorder=12)

            ax_img.axhline(cy - 1.5, color='crimson', linestyle=':', linewidth=1.5)
            ax_img.axhline(cy + 1.5, color='crimson', linestyle=':', linewidth=1.5)
            ax_img.axvline(cx - 1.5, color='dodgerblue', linestyle=':', linewidth=1.5)
            ax_img.axvline(cx + 1.5, color='dodgerblue', linestyle=':', linewidth=1.5)

            ax_img.set_xlabel(r"Right Ascension (J2000)", size=20)
            ax_img.set_ylabel(r"Declination (J2000)", size=20, labelpad=1)

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

            blob_label = f"Peak {b_idx + 1}" if "2peak" in fittype else (f"Blob {b_idx + 1}" if len(advi_coords) > 1 else "Blob")
            fig.suptitle(f"{target_name} - {blob_label} Zoom ($2'' \\times 2''$)", size=26)
            fig.subplots_adjust(hspace=0.2, wspace=0.28, bottom=0.15)
            pp.savefig(fig)
            plt.close(fig)

    # --------------------------------------------------------------------------
    # 8. Publication-Grade Parameter Summary Table
    # --------------------------------------------------------------------------
    render_summary_table_page(samples, pars_map, fittype, pp)

    pp.close()
    logging.info(f"Successfully generated all diagnostic figures in {pdf_plot}")
    logging.info("ADVI Post-Processing Pipeline Completed.")


if __name__ == '__main__':
    main()
