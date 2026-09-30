r"""
Unit and Integration Tests for ADVI Visibility Fitting Pipeline
================================================================
Tests coordinate transformations, analytical gradients, prior Jacobians,
reparameterizations, and variational ELBO optimization.

Usage:
    python test_advi.py
"""

import sys
import os
import unittest
import numpy as np

# Ensure current directory is on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Gracefully mock galario if running in a lightweight local test environment without GPU/C++ libraries
from unittest.mock import MagicMock
try:
    import galario
except ImportError:
    mock_galario = MagicMock()
    mock_galario_double = MagicMock()
    mock_galario_double.deg = np.pi / 180.0
    mock_galario_double.arcsec = np.pi / (180.0 * 3600.0)
    mock_galario_double.chi2Image = lambda image, dxy, u, v, Re, Im, w, **kwargs: float(np.sum(w * (np.sum(image) - Re)**2 + w * Im**2))
    mock_galario_double.sampleImage = lambda image, dxy, u, v, **kwargs: np.full_like(u, np.sum(image), dtype=np.complex128)
    mock_galario_double.get_image_size = lambda u, v, **kwargs: (512, 0.005 * (np.pi / (180.0 * 3600.0)))
    sys.modules['galario'] = mock_galario
    sys.modules['galario.double'] = mock_galario_double

from prior_advi import (
    theta_to_zeta,
    zeta_to_theta,
    grad_zeta_to_theta,
    log_prior_and_jacobian,
    grad_log_prior_and_jacobian,
    get_prior_bounds,
    sample_prior,
    sigmoid,
    logit,
    log_sigmoid_prod
)
from model_prof import model_addon, get_grid, gaussring_prof, gaussblob_prof, ring_flux_to_peak, blob_flux_to_peak, model_prof
from run_advi import AdviVisibilityFitter
import run_advi


class TestAdviPrimitives(unittest.TestCase):
    """
    Tests numerical stability of sigmoid, logit, and log_sigmoid_prod.
    """
    def test_sigmoid_and_logit_roundtrip(self):
        u_vals = np.array([1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0 - 1e-6])
        zeta = logit(u_vals)
        u_rec = sigmoid(zeta)
        np.testing.assert_allclose(u_rec, u_vals, rtol=1e-5, atol=1e-8)

    def test_sigmoid_extreme_values(self):
        z_large_pos = np.array([50.0, 100.0, 500.0])
        z_large_neg = np.array([-50.0, -100.0, -500.0])
        np.testing.assert_allclose(sigmoid(z_large_pos), 1.0, atol=1e-15)
        np.testing.assert_allclose(sigmoid(z_large_neg), 0.0, atol=1e-15)

    def test_log_sigmoid_prod_numerical_stability(self):
        z = np.array([-100.0, -10.0, 0.0, 10.0, 100.0])
        val = log_sigmoid_prod(z)
        self.assertTrue(np.all(np.isfinite(val)))
        self.assertTrue(np.all(val <= -1.386))


class TestAdviTransformations(unittest.TestCase):
    """
    Tests unconstrained coordinate transformations and Jacobians for both models.
    """
    def test_roundtrip_twod_gaussring(self):
        fittype = 'twod_gaussring'
        np.random.seed(42)

        for _ in range(20):
            theta_orig = sample_prior(fittype)
            zeta = theta_to_zeta(theta_orig, fittype)
            theta_rec = zeta_to_theta(zeta, fittype)
            np.testing.assert_allclose(theta_rec, theta_orig, rtol=1e-5, atol=1e-7,
                                       err_msg="Roundtrip theta -> zeta -> theta failed for twod_gaussring")

    def test_roundtrip_twod_gauss1blob(self):
        fittype = 'twod_gauss1blob'
        np.random.seed(42)

        for _ in range(20):
            theta_orig = sample_prior(fittype)
            zeta = theta_to_zeta(theta_orig, fittype)
            theta_rec = zeta_to_theta(zeta, fittype)
            np.testing.assert_allclose(theta_rec, theta_orig, rtol=1e-5, atol=1e-7,
                                       err_msg="Roundtrip theta -> zeta -> theta failed for twod_gauss1blob")

    def test_grad_zeta_to_theta(self):
        for fittype in ('twod_gaussring', 'twod_gauss1blob'):
            _, _, ndim = model_addon(fittype)
            np.random.seed(123)
            zeta = np.random.normal(0.0, 1.0, size=ndim)

            ana_grad = grad_zeta_to_theta(zeta, fittype)

            h = 1e-6
            num_grad = np.empty(ndim)
            for j in range(ndim):
                zp = zeta.copy()
                zm = zeta.copy()
                zp[j] += h
                zm[j] -= h
                tp = zeta_to_theta(zp, fittype)
                tm = zeta_to_theta(zm, fittype)
                num_grad[j] = (tp[j] - tm[j]) / (2.0 * h)

            np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-4, atol=1e-6,
                                       err_msg=f"Analytical dtheta/dzeta mismatch in {fittype}")

    def test_grad_log_prior_and_jacobian(self):
        for fittype in ('twod_gaussring', 'twod_gauss1blob'):
            _, _, ndim = model_addon(fittype)
            np.random.seed(456)
            zeta = np.random.normal(0.0, 1.5, size=ndim)

            ana_grad = grad_log_prior_and_jacobian(zeta, fittype)

            h = 1e-6
            num_grad = np.empty(ndim)
            for j in range(ndim):
                zp = zeta.copy()
                zm = zeta.copy()
                zp[j] += h
                zm[j] -= h
                lp_p = log_prior_and_jacobian(zp, fittype)
                lp_m = log_prior_and_jacobian(zm, fittype)
                num_grad[j] = (lp_p - lp_m) / (2.0 * h)

            np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-4, atol=1e-6,
                                       err_msg=f"Analytic prior gradient mismatch in {fittype}")


class TestModelProfileMetadata(unittest.TestCase):
    """
    Tests model definitions and metadata restrictions.
    """
    def test_supported_models(self):
        l7, u7, ndim7 = model_addon('twod_gaussring')
        self.assertEqual(ndim7, 7)
        self.assertEqual(len(l7), 7)
        self.assertEqual(len(u7), 7)

        l11, u11, ndim11 = model_addon('twod_gauss1blob')
        self.assertEqual(ndim11, 11)
        self.assertEqual(len(l11), 11)
        self.assertEqual(len(u11), 11)

    def test_unsupported_models_rejected(self):
        with self.assertRaises(ValueError):
            model_addon('twod_gauss2blob')
        with self.assertRaises(ValueError):
            model_addon('simgauss')


class TestAdviOptimization(unittest.TestCase):
    """
    Integration tests for ADVI optimization loop and posterior extraction.
    """
    def setUp(self):
        # Create small synthetic visibility dataset
        np.random.seed(789)
        u = np.linspace(1e3, 1e5, 30)
        v = np.linspace(1e3, 1e5, 30)
        re = np.ones(30) * 0.05
        im = np.zeros(30)
        w = np.ones(30) * 50.0
        self.vis_data = (u, v, re, im, w)
        self.args = (64, 0.01 * np.pi / (180.0 * 3600.0))
        run_advi.GLOBAL_DATA = self.vis_data

    def test_fit_twod_gaussring(self):
        fitter = AdviVisibilityFitter(
            fittype='twod_gaussring',
            args=self.args,
            vis_data=self.vis_data,
            pool=None,
            eta=0.05,
            num_samples=2,
            tol_rel_obj=0.001,
            seed=42
        )
        fitter.fit(max_iters=5)
        self.assertEqual(len(fitter.elbo_history), 5)
        self.assertTrue(np.all(np.isfinite(fitter.elbo_history)))

        samples, best_fit = fitter.sample_posterior(num_draws=20)
        self.assertEqual(samples.shape, (20, 7))
        self.assertEqual(len(best_fit), 7)
        self.assertTrue(np.all(np.isfinite(samples)))
        self.assertTrue(np.all(np.isfinite(best_fit)))

    def test_fit_twod_gauss1blob(self):
        fitter = AdviVisibilityFitter(
            fittype='twod_gauss1blob',
            args=self.args,
            vis_data=self.vis_data,
            pool=None,
            eta=0.05,
            num_samples=2,
            tol_rel_obj=0.001,
            seed=42
        )
        fitter.fit(max_iters=5)
        self.assertEqual(len(fitter.elbo_history), 5)
        self.assertTrue(np.all(np.isfinite(fitter.elbo_history)))

        samples, best_fit = fitter.sample_posterior(num_draws=20)
        self.assertEqual(samples.shape, (20, 11))
        self.assertEqual(len(best_fit), 11)
        self.assertTrue(np.all(np.isfinite(samples)))
        self.assertTrue(np.all(np.isfinite(best_fit)))


if __name__ == '__main__':
    unittest.main()
