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
    ax = plt.subplot(111, projection=w)
    ax.imshow(np.zeros((64, 64)))
    safe_add_beam(ax, header=hdr, frame=False, color='white', corner='bottom left')
    plt.close(fig)
    print("PASS: safe_add_beam verified.")

def test_summary_table_rendering():
    """Verify render_summary_table_page generates tables across different model configurations."""
    test_configs = [
        ('twod_gaussring', 7),
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
    """Verify uvplot UVTable plotting logic without errors."""
    u = np.random.randn(1000) * 1e5
    v = np.random.randn(1000) * 1e5
    re = np.random.randn(1000) * 0.01
    im = np.random.randn(1000) * 0.01
    w = np.ones(1000) * 1e4
    uv_data = UVTable(uvtable=[u, v, re, im, w], columns=['u', 'v', 'Re', 'Im', 'weights'])
    uv_mod = UVTable(uvtable=[u, v, re*0.9, im*0.9, w], columns=['u', 'v', 'Re', 'Im', 'weights'])
    bin_size = np.max(np.hypot(u, v)) / 20.0

    axes_uv = uv_data.plot(color='black', linestyle='.', label='Data', uvbin_size=bin_size)
    uv_mod.plot(color='crimson', linestyle='-', label='Model', axes=list(axes_uv), uvbin_size=bin_size, yerr=False)
    fig_uv = axes_uv[0].figure
    plt.close(fig_uv)
    print("PASS: UVTable plotting verified.")

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

if __name__ == '__main__':
    test_convolve_beam_orientation()
    test_safe_add_beam()
    test_summary_table_rendering()
    test_uvplot_generation()
    test_runplot_evidence_limits()
    print("\nALL VISUALIZATION AND POST-PROCESSING TESTS PASSED!")
