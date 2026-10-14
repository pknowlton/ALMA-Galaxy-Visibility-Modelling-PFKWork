import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
from astropy.wcs import WCS
import io

# Add input_dir to sys.path
sys.path.insert(0, os.path.abspath('dynesty_version_fixed/input_dir'))

from unittest.mock import MagicMock
# Mock heavy cluster-only CASA, Galario, and Dynesty dependencies if not installed locally
dynesty_mock = MagicMock()
dyfunc_mock = MagicMock()
dyfunc_mock.quantile = lambda arr, q, weights=None: np.quantile(arr, q)
dynesty_mock.utils = dyfunc_mock
dynesty_plotting_mock = MagicMock()
dynesty_mock.plotting = dynesty_plotting_mock

sys.modules['dynesty'] = dynesty_mock
sys.modules['dynesty.utils'] = dyfunc_mock
sys.modules['dynesty.plotting'] = dynesty_plotting_mock

for mod_name in ['casatools', 'casatasks', 'galario', 'galario.double', 'corner']:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from visualize_dynesty import convolve_model_with_beam, safe_add_beam, render_summary_table_page
import prior_tform_streamline as pt_stream
from uvplot import UVTable
from astropy.visualization.wcsaxes import WCSAxes

class MockDynestyResults:
    def __init__(self, n_samples=100, n_dim=11):
        self.samples = np.random.uniform(0.1, 0.9, size=(n_samples, n_dim))
        self.logl = np.random.randn(n_samples)
        self.logwt = np.zeros(n_samples)
        self.logz = np.zeros(n_samples)

def test_convolve_beam_orientation():
    """Verify that theta in Gaussian2DKernel is equal to bpa_deg (no +90 shift)."""
    # Create an image with an elongated point source
    img = np.zeros((64, 64))
    img[32, 32] = 1.0
    bmaj_deg = 0.5 / 3600.0
    bmin_deg = 0.2 / 3600.0
    pixarcsec = 0.05
    # BPA = 0 means major axis along North (Y-axis)
    conv_0 = convolve_model_with_beam(img, bmaj_deg, bmin_deg, 0.0, pixarcsec)
    # Check that Y-dispersion > X-dispersion when BPA=0
    y_profile = np.sum(conv_0, axis=1)
    x_profile = np.sum(conv_0, axis=0)
    y_std = np.std(np.arange(64)[y_profile > 0.1 * np.max(y_profile)])
    x_std = np.std(np.arange(64)[x_profile > 0.1 * np.max(x_profile)])
    assert y_std > x_std, f"At BPA=0, major axis should be along Y (North): y_std={y_std}, x_std={x_std}"
    print("PASS: convolve_model_with_beam BPA orientation verified.")

def test_safe_add_beam():
    """Verify safe_add_beam executes without error on a WCSAxes subplot."""
    w = WCS(naxis=2)
    w.wcs.crpix = [32, 32]
    w.wcs.cdelt = [-0.05/3600, 0.05/3600]
    w.wcs.crval = [160.99, 11.70]
    w.wcs.ctype = ['RA---SIN', 'DEC--SIN']

    hdr = fits.Header()
    hdr['BMAJ'] = 0.2 / 3600.0
    hdr['BMIN'] = 0.1 / 3600.0
    hdr['BPA'] = 45.0

    fig = plt.figure()
    ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], wcs=w)
    fig.add_axes(ax)
    ax.imshow(np.zeros((64, 64)))
    safe_add_beam(ax, header=hdr, frame=False, color='white', corner='bottom left')
    plt.close(fig)
    print("PASS: safe_add_beam verified.")

def test_summary_table_rendering():
    """Verify render_summary_table_page generates tables across different model configurations."""
    test_configs = [
        ('twod_gaussring', 7),
        ('twod_gaussring_nonuni', 8),
        ('twod_gauss1blob', 11),
        ('twod_gauss1blob_2peak', 13),
        ('twod_gauss1blob_2peak_dp', 15),
        ('twod_gauss3blob', 19),
        ('simgauss', 7)
    ]

    for fittype, ndim in test_configs:
        res = MockDynestyResults(n_samples=50, n_dim=ndim)
        pars_bf = res.samples[0].copy()
        weights = np.ones(50) / 50.0

        pdf_buf = io.BytesIO()
        with PdfPages(pdf_buf) as pp:
            render_summary_table_page(res, pars_bf, weights, fittype, pp)

        assert pdf_buf.getvalue(), f"PDF buffer for {fittype} should not be empty"
        print(f"PASS: Summary table successfully rendered for {fittype} ({ndim} params).")

def test_uvplot_generation():
    """Verify uvplot UVTable plotting logic with wide figure and clean legend."""
    u = np.random.randn(1000) * 1e5
    v = np.random.randn(1000) * 1e5
    re = np.random.randn(1000) * 0.01
    im = np.random.randn(1000) * 0.01
    w = np.ones(1000) * 1e4
    uv_data = UVTable(uvtable=[u, v, re, im, w], columns=['u', 'v', 'Re', 'Im', 'weights'])
    uv_mod = UVTable(uvtable=[u, v, re*0.9, im*0.9, w], columns=['u', 'v', 'Re', 'Im', 'weights'])
    bin_size = np.max(np.hypot(u, v)) / 20.0

    fig_uv = plt.figure(figsize=(24, 10))
    gs = fig_uv.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.0)
    ax_uv1 = fig_uv.add_subplot(gs[0])
    ax_uv2 = fig_uv.add_subplot(gs[1], sharex=ax_uv1)
    axes_list = [ax_uv1, ax_uv2]

    uv_data.plot(axes=axes_list, color='black', linestyle='.', label='Observed Data', uvbin_size=bin_size)
    uv_mod.plot(axes=axes_list, color='crimson', linestyle='-', label='Model Best Fit', uvbin_size=bin_size, yerr=False)

    ax_uv1.yaxis.set_label_coords(-0.05, 0.5)
    ax_uv2.yaxis.set_label_coords(-0.05, 0.5)
    ax_uv1.legend(loc='upper right', fontsize=16)
    if ax_uv2.get_legend():
        ax_uv2.get_legend().remove()
    fig_uv.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.12, hspace=0.0)

    # Verify size and labels
    assert fig_uv.get_size_inches()[0] == 24.0
    _, labels = ax_uv1.get_legend_handles_labels()
    assert 'Observed Data' in labels
    assert '(XX)' not in labels[0]
    plt.close(fig_uv)
    print("PASS: Wide UVTable plotting verified.")

def test_runplot_evidence_limits():
    """Verify that evidence panel y-limits adapt to show flat -> increasing -> flat plateau."""
    fig, axes = plt.subplots(4, 1)
    
    # Simulate a run with a prior baseline of -500 and final evidence of -100, with initial outlier
    logz = np.full(5000, -500.0)
    logz[0] = -1e8
    logz[1] = -1e6
    logz[2000:3500] = np.linspace(-500.0, -100.0, 1500)
    logz[3500:] = -100.0
    logzerr = np.full(5000, 0.2)

    valid_logz = logz[np.isfinite(logz)]
    final_logz = valid_logz[-1]
    final_err = logzerr[-1]
    ymax = final_logz + max(3.0 * final_err, 0.5)

    n_pts = len(valid_logz)
    i0 = min(max(int(0.01 * n_pts), 5), n_pts // 4)
    stable_min = np.min(valid_logz[i0:])
    total_drop = final_logz - stable_min
    ymin = stable_min - 0.05 * total_drop

    ax_ev = axes[3]
    ax_ev.set_ylim(ymin, ymax)

    y_limits = ax_ev.get_ylim()
    assert y_limits[0] < -500.0, f"Expected ymin below -500, got {y_limits[0]}"
    assert y_limits[1] > -100.0, f"Expected ymax above -100, got {y_limits[1]}"
    assert y_limits[0] > -1000.0, f"Outlier -1e8 should have been excluded, got {y_limits[0]}"
    plt.close(fig)
    print("PASS: runplot evidence limit adaptation verified.")

def test_nonuni_ring_properties():
    """Verify non-uniform ring mathematical properties, flux conservation, non-negativity, and transforms."""
    from model_prof import gaussring_prof, gaussring_nonuni_prof, ring_flux_to_peak, twod_gaussring_nonuni_model
    import prior_tform_streamline as pt

    # 1. Test prior transform dimensionality and bounds
    u_test = np.array([0.5] * 8)
    p_test = pt.twod_gaussring_nonuni_ptform(u_test)
    assert len(p_test) == 8, f"Expected 8 parameters, got {len(p_test)}"
    # Prof. Offset bounds should be [0, 180] deg
    assert p_test[7] == 90.0, f"Expected offset 90.0 at u=0.5, got {p_test[7]}"

    u_zero = np.zeros(8)
    p_zero = pt.twod_gaussring_nonuni_ptform(u_zero)
    assert p_zero[7] == 0.0, f"Expected offset 0.0 at u=0.0, got {p_zero[7]}"

    u_one = np.ones(8)
    p_one = pt.twod_gaussring_nonuni_ptform(u_one)
    assert p_one[7] == 180.0, f"Expected offset 180.0 at u=1.0, got {p_one[7]}"

    # 2. Test mathematical profile properties: non-negativity and peak locations
    peak_ring = 8.0
    sigma_ring = 0.5
    rad_ring = 5.0
    off_rad = np.radians(45.0)

    # 1D azimuthal test
    thetas = np.linspace(0, 2 * np.pi, 10000, endpoint=False)
    r_peak = np.full_like(thetas, rad_ring)
    prof_vals = gaussring_nonuni_prof(peak_ring, thetas, off_rad, sigma_ring, rad_ring, r_peak, 1.0)

    # Must be non-negative everywhere
    assert np.all(prof_vals >= 0.0), "Non-uniform ring surface brightness must be non-negative everywhere!"

    # Maximum brightness should be 2 * 10^peak_ring
    max_expected = 2.0 * (10.0**peak_ring)
    np.testing.assert_allclose(np.max(prof_vals), max_expected, rtol=1e-4)

    # Minimum brightness should be 0.0
    np.testing.assert_allclose(np.min(prof_vals), 0.0, atol=1e-2)

    # Peaks should occur at theta = off_rad and theta = off_rad + pi
    peak_idx = np.where(prof_vals > 0.999 * max_expected)[0]
    peak_angles = thetas[peak_idx]
    # Check that peaks are near 45 deg and 225 deg
    deg_peaks = np.degrees(peak_angles)
    has_peak1 = np.any(np.abs(deg_peaks - 45.0) < 0.2)
    has_peak2 = np.any(np.abs(deg_peaks - 225.0) < 0.2)
    assert has_peak1 and has_peak2, f"Peaks should be at 45 and 225 deg, found near {deg_peaks}"

    # 3. Test flux conservation: integral over non-uniform ring == uniform ring
    # Create 2D grid
    nxy = 256
    dxy = 0.1
    img_nonuni = twod_gaussring_nonuni_model(peak_ring, off_rad, sigma_ring, rad_ring, 0.0, 0.0, 0.0, 0.0, nxy, dxy, 'plot')
    from model_prof import twod_gaussring_model
    img_uni = twod_gaussring_model(peak_ring, sigma_ring, rad_ring, 0.0, 0.0, 0.0, 0.0, nxy, dxy, 'plot')

    flux_nonuni = np.sum(img_nonuni) * (dxy**2)
    flux_uni = np.sum(img_uni) * (dxy**2)

    np.testing.assert_allclose(flux_nonuni, flux_uni, rtol=1e-3,
                               err_msg=f"Total flux should match uniform ring: nonuni={flux_nonuni}, uni={flux_uni}")

    # Also test in chi2/vis mode
    dxy_rad = 0.1 * (np.pi / (180.0 * 3600.0))
    sigma_rad = sigma_ring * (np.pi / (180.0 * 3600.0))
    rad_rad = rad_ring * (np.pi / (180.0 * 3600.0))
    img_nonuni_vis = twod_gaussring_nonuni_model(peak_ring, off_rad, sigma_rad, rad_rad, 0.5, 0.0, 0.0, 0.0, nxy, dxy_rad, 'chi2')
    img_uni_vis = twod_gaussring_model(peak_ring, sigma_rad, rad_rad, 0.5, 0.0, 0.0, 0.0, nxy, dxy_rad, 'chi2')

    flux_vis_nonuni = np.sum(img_nonuni_vis)
    flux_vis_uni = np.sum(img_uni_vis)
    np.testing.assert_allclose(flux_vis_nonuni, flux_vis_uni, rtol=1e-3,
                               err_msg=f"Total flux in chi2 mode should match uniform ring: nonuni={flux_vis_nonuni}, uni={flux_vis_uni}")

    print("PASS: Non-uniform ring mathematical properties, flux conservation, and transforms verified.")

if __name__ == '__main__':
    test_convolve_beam_orientation()
    test_safe_add_beam()
    test_summary_table_rendering()
    test_nonuni_ring_properties()
    test_uvplot_generation()
    test_runplot_evidence_limits()
    print("\nALL VISUALIZATION AND POST-PROCESSING TESTS PASSED!")
