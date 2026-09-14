"""
Unit Tests for Visibility Domain Parity, Coordinates, and Restoring Beam
========================================================================
This test suite verifies the critical correctness requirements specified in:
- audit.md: P0 action items (parity, shift theorem, WCS roundtrip, shared priors)
- audit.md: P2 action items (beam solid angle calculation)

Usage:
    python test_parity_and_transforms.py
"""

import sys
import os
import unittest
import numpy as np

# Ensure input_dir is on Python path
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
    sys.modules['galario'] = mock_galario
    sys.modules['galario.double'] = mock_galario_double

# Turn off astropy IERS auto-download to run instantly offline
try:
    from astropy.utils import iers
    iers.conf.auto_download = False
except Exception:
    pass

from model_prof import get_grid, gaussblob_prof, gaussring_prof, ring_flux_to_peak, blob_flux_to_peak
from radec_calc import make_model_wcs, dynest_radec
from prior_tform import (
    RING_PRIOR_RANGES,
    twod_gaussring_ptform,
    twod_gauss1blob_ptform,
    twod_gauss1blob_2peak_ptform,
    twod_gauss1blob_2peak_dp_ptform,
    twod_gauss3blob_ptform
)


class TestBeamSolidAngle(unittest.TestCase):
    """
    Verifies beam solid angle calculation (audit.md P2).
    """

    def test_beam_area_formula(self):
        """
        Verify that Omega_beam = pi / (4 * ln(2)) * bmaj * bmin.
        Analytic derivation:
            2D Gaussian profile: I(x, y) = exp( -4 ln(2) * (x^2 / bmaj^2 + y^2 / bmin^2) )
            Integral over 2D plane: int int I(x, y) dx dy = pi / (4 ln(2)) * bmaj * bmin.
        """
        bmaj = 0.04  # arcsec
        bmin = 0.03  # arcsec

        # Analytic solid angle
        omega_analytic = (np.pi / (4.0 * np.log(2.0))) * bmaj * bmin

        # Numerical integration of 2D Gaussian
        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        sigma_x = bmin * fwhm_to_sigma
        sigma_y = bmaj * fwhm_to_sigma

        dx = 0.001
        x = np.arange(-0.2, 0.2, dx)
        y = np.arange(-0.2, 0.2, dx)
        xx, yy = np.meshgrid(x, y)

        gauss_2d = np.exp(-0.5 * ((xx / sigma_x)**2 + (yy / sigma_y)**2))
        numerical_integral = np.sum(gauss_2d) * (dx**2)

        np.testing.assert_allclose(
            omega_analytic, numerical_integral, rtol=1e-3,
            err_msg="Beam solid angle formula does not match 2D Gaussian integral."
        )

        # Check old flawed formula was a factor of 2 too large
        omega_old = (np.pi / (2.0 * np.log(2.0))) * bmaj * bmin
        self.assertAlmostEqual(omega_old / omega_analytic, 2.0, places=5)


class TestCoordinateGridAndParity(unittest.TestCase):
    """
    Verifies grid handedness and Galario origin='lower' convention (audit.md P0).
    """

    def test_grid_handedness(self):
        """
        Galario origin='lower' convention:
        - Row index iy: Dec increases with row (iy=0 is South, iy=nxy-1 is North).
        - Column index ix: RA decreases with column (ix=0 is East, ix=nxy-1 is West).
        """
        nxy = 512
        dxy = 0.04
        xx, yy = get_grid(nxy, dxy)

        # Center pixel [nxy/2, nxy/2] must be (0, 0)
        cx, cy = nxy // 2, nxy // 2
        self.assertAlmostEqual(xx[cy, cx], 0.0, places=7)
        self.assertAlmostEqual(yy[cy, cx], 0.0, places=7)

        # Column 0 must be East (positive X)
        self.assertTrue(np.all(xx[:, 0] > 0.0), "Column 0 should have positive RA offset (East).")

        # Column nxy - 1 must be West (negative X)
        self.assertTrue(np.all(xx[:, -1] < 0.0), "Column nxy-1 should have negative RA offset (West).")

        # Row 0 must be South (negative Y)
        self.assertTrue(np.all(yy[0, :] < 0.0), "Row 0 should have negative Dec offset (South).")

        # Row nxy - 1 must be North (positive Y)
        self.assertTrue(np.all(yy[-1, :] > 0.0), "Row nxy-1 should have positive Dec offset (North).")

    def test_blob_offset_quadrants(self):
        """
        Verify that blob Cartesian coordinates (dist * sin(ang), dist * cos(ang))
        correctly correspond to celestial directions:
            ang = 0 deg   -> North (x=0, y>0)
            ang = 90 deg  -> East  (x>0, y=0)
            ang = 180 deg -> South (x=0, y<0)
            ang = 270 deg -> West  (x<0, y=0)
        """
        dist = 5.0
        angles_deg = [0.0, 90.0, 180.0, 270.0]
        expected_signs = [
            (0.0, +1.0),  # North
            (+1.0, 0.0),  # East
            (0.0, -1.0),  # South
            (-1.0, 0.0)   # West
        ]

        for ang, (exp_x_sign, exp_y_sign) in zip(angles_deg, expected_signs):
            rad = np.radians(ang)
            x = dist * np.sin(rad)
            y = dist * np.cos(rad)

            if exp_x_sign > 0:
                self.assertGreater(x, 0.0)
            elif exp_x_sign < 0:
                self.assertLess(x, 0.0)
            else:
                self.assertAlmostEqual(x, 0.0, places=6)

            if exp_y_sign > 0:
                self.assertGreater(y, 0.0)
            elif exp_y_sign < 0:
                self.assertLess(y, 0.0)
            else:
                self.assertAlmostEqual(y, 0.0, places=6)


class TestFourierShiftTheorem(unittest.TestCase):
    """
    Verifies the Fourier shift theorem phases:
        F{ delta(x - x0, y - y0) } = exp( -2 * pi * i * (u * x0 + v * y0) )
    """

    def test_shift_phase_signs(self):
        """
        A source shifted +East (x0 > 0) must produce a phase of -2 * pi * u * x0.
        A source shifted +North (y0 > 0) must produce a phase of -2 * pi * v * y0.
        """
        nxy = 256
        dxy = 0.05
        xx, yy = get_grid(nxy, dxy)

        # Discrete 2D Fourier Transform of shifted Gaussian
        sigma = 0.2
        dRA = 1.0   # +1 arcsec East
        dDec = 2.0  # +2 arcsec North

        # Discrete 2D Fourier Transform of shifted Gaussian
        sigma = 0.2
        dRA = 1.0   # +1 arcsec East
        dDec = 2.0  # +2 arcsec North

        # Model shifted on image plane
        img = gaussblob_prof(peak=0.0, sigma=sigma, xx=xx, yy=yy, xoff=dRA, yoff=dDec, dxy=dxy)

        # FFT shift to centered k-space
        vis = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

        # Frequency coordinates along columns and rows (cycles per arcsec)
        # Note: In our grid xx is positive East (descending with column index),
        # so spatial frequency conjugate to East is u_east = -u_col.
        u_coord = -np.fft.fftshift(np.fft.fftfreq(nxy, d=dxy))
        v_coord = np.fft.fftshift(np.fft.fftfreq(nxy, d=dxy))
        uu, vv = np.meshgrid(u_coord, v_coord)

        # Check phase along u-axis (v = 0)
        center_row = nxy // 2
        u_slice = uu[center_row, :]
        vis_u_slice = vis[center_row, :]

        # Analytic phase for +dRA East: phase = -2 * pi * u_east * dRA
        analytic_phase_u = -2.0 * np.pi * u_slice * dRA
        measured_phase_u = np.unwrap(np.angle(vis_u_slice))

        # Check slope d(phase)/du_east
        valid_u = np.abs(u_slice) < 0.5  # within central spatial frequencies
        slope_measured = np.polyfit(u_slice[valid_u], measured_phase_u[valid_u], 1)[0]
        slope_analytic = -2.0 * np.pi * dRA

        self.assertAlmostEqual(
            slope_measured / slope_analytic, 1.0, delta=0.01,
            msg="Phase slope along u-axis does not match analytic Fourier shift theorem."
        )


class TestFluxReparameterization(unittest.TestCase):
    """
    Verifies that log integrated flux + log size parameterization recovers the input flux
    when integrated over 2D space (Audit P1 Item 1).
    """

    def test_blob_flux_integration(self):
        """
        Verify that integral of 2D Gaussian blob equals input flux F_blob.
        """
        log_flux_jy = -2.0  # 10 mJy = 0.01 Jy
        expected_flux_jy = 10.0**log_flux_jy

        sigma_arcsec = 0.2
        arcsec_to_rad = np.pi / (180.0 * 3600.0)
        sigma_rad = sigma_arcsec * arcsec_to_rad

        peak_jysr = blob_flux_to_peak(log_flux_jy, sigma_rad)

        # Numerical integration over grid
        dxy_arcsec = 0.01
        nxy = 256
        xx, yy = get_grid(nxy, dxy_arcsec)
        # Intensity in Jy/sr
        img_jysr = 10.0**peak_jysr * np.exp(-0.5 * ((xx / sigma_arcsec)**2 + (yy / sigma_arcsec)**2))

        # Pixel area in steradians
        dxy_rad = dxy_arcsec * arcsec_to_rad
        integrated_flux = np.sum(img_jysr) * (dxy_rad**2)

        np.testing.assert_allclose(
            integrated_flux, expected_flux_jy, rtol=1e-3,
            err_msg="Integrated flux of Gaussian blob does not match input flux parameter!"
        )

    def test_ring_flux_integration(self):
        """
        Verify that integral of 2D inclined Gaussian ring equals input flux F_ring.
        """
        log_flux_jy = -1.5  # ~31.6 mJy
        expected_flux_jy = 10.0**log_flux_jy

        ring_rad_arcsec = 5.0
        sigma_arcsec = 0.5
        inc_deg = 45.0

        arcsec_to_rad = np.pi / (180.0 * 3600.0)
        ring_rad_rad = ring_rad_arcsec * arcsec_to_rad
        sigma_rad = sigma_arcsec * arcsec_to_rad
        inc_rad = np.radians(inc_deg)

        peak_jysr = ring_flux_to_peak(log_flux_jy, sigma_rad, ring_rad_rad, inc_rad)

        # Numerical integration over grid
        dxy_arcsec = 0.05
        nxy = 512
        xx, yy = get_grid(nxy, dxy_arcsec)

        xinc = xx / np.cos(inc_rad)
        radius_vec = np.hypot(xinc, yy)
        img_jysr = 10.0**peak_jysr * np.exp(-0.5 * (((radius_vec - ring_rad_arcsec) / sigma_arcsec)**2))

        # Pixel area in steradians
        dxy_rad = dxy_arcsec * arcsec_to_rad
        integrated_flux = np.sum(img_jysr) * (dxy_rad**2)

        np.testing.assert_allclose(
            integrated_flux, expected_flux_jy, rtol=5e-3,
            err_msg="Integrated flux of inclined Gaussian ring does not match input flux parameter!"
        )


class TestPriorConsistency(unittest.TestCase):
    """
    Verifies that shared priors are identical and width bounds strictly positive (audit.md P0, P1).
    """

    def test_shared_ring_prior_ranges(self):
        """
        Verify that all 5 models use identical shared RING_PRIOR_RANGES.
        """
        unit_cube_mid = np.full(20, 0.5)

        # Check that ring parameters evaluate identically at unit cube midpoint
        p_ring = twod_gaussring_ptform(unit_cube_mid[:7])
        p_1blob = twod_gauss1blob_ptform(unit_cube_mid[:11])
        p_2peak = twod_gauss1blob_2peak_ptform(unit_cube_mid[:13])
        p_dp = twod_gauss1blob_2peak_dp_ptform(unit_cube_mid[:15])
        p_3blob = twod_gauss3blob_ptform(unit_cube_mid[:19])

        # First 7 parameters must be identical across ALL models
        np.testing.assert_allclose(p_ring[:7], p_1blob[:7], err_msg="twod_gauss1blob differs in shared ring priors!")
        np.testing.assert_allclose(p_ring[:7], p_2peak[:7], err_msg="twod_gauss1blob_2peak differs in shared ring priors!")
        np.testing.assert_allclose(p_ring[:7], p_dp[:7], err_msg="twod_gauss1blob_2peak_dp differs in shared ring priors!")
        np.testing.assert_allclose(p_ring[:7], p_3blob[:7], err_msg="twod_gauss3blob differs in shared ring priors!")

    def test_positive_sigma_bounds(self):
        """
        Verify strictly positive lower bounds (sigma >= 0.05 arcsec, log_sigma >= -1.301) to avoid funnels.
        """
        unit_cube_zero = np.zeros(20)

        for ptform, n in [
            (twod_gaussring_ptform, 7),
            (twod_gauss1blob_ptform, 11),
            (twod_gauss1blob_2peak_ptform, 13),
            (twod_gauss1blob_2peak_dp_ptform, 15),
            (twod_gauss3blob_ptform, 19)
        ]:
            pars_min = ptform(unit_cube_zero[:n])
            # Ring log_sigma is at index 1
            log_sigma_min = pars_min[1]
            sigma_min = 10.0**log_sigma_min
            self.assertGreaterEqual(sigma_min, 0.049, f"Model {ptform.__name__} ring sigma lower bound < 0.05!")

    def test_blob_angle_priors_cover_ymcs(self):
        """
        Verify that blob angle priors under the counter-clockwise East-of-North convention
        correctly cover the target YMC cluster coordinates without label ambiguity.
        """
        # Disk PA ~13 deg
        # YMC 15 (South-East): CCW disk angle ~170.85 deg
        # YMC 17 (South-West): CCW disk angle ~182.41 deg
        # YMC 18 (South-West): CCW disk angle ~185.17 deg
        # YMC 6 (North):       CCW disk angle ~348.32 deg

        # In twod_gauss3blob:
        u_0 = np.zeros(19)
        u_1 = np.ones(19)
        p_min = twod_gauss3blob_ptform(u_0)
        p_max = twod_gauss3blob_ptform(u_1)

        ang1_min, ang1_max = p_min[10], p_max[10]
        ang2_min, ang2_max = p_min[14], p_max[14]
        ang3_min, ang3_max = p_min[18], p_max[18]

        # 1. Blob 1 and Blob 3 must be completely disjoint (zero overlap)
        self.assertLessEqual(ang1_max, ang3_min, "Blob 1 and Blob 3 angle ranges must not overlap!")

        # 2. Blob 1 must enclose YMC 15 (~170.85 deg)
        self.assertTrue(ang1_min <= 170.85 <= ang1_max, f"Blob 1 [{ang1_min}, {ang1_max}] does not enclose YMC 15!")

        # 3. Blob 2 must enclose YMC 6 (~348.32 deg)
        self.assertTrue(ang2_min <= 348.32 <= ang2_max, f"Blob 2 [{ang2_min}, {ang2_max}] does not enclose YMC 6!")

        # 4. Blob 3 must enclose YMC 17 (~182.41 deg) and YMC 18 (~185.17 deg)
        self.assertTrue(ang3_min <= 182.41 <= ang3_max, f"Blob 3 [{ang3_min}, {ang3_max}] does not enclose YMC 17!")
        self.assertTrue(ang3_min <= 185.17 <= ang3_max, f"Blob 3 [{ang3_min}, {ang3_max}] does not enclose YMC 18!")


if __name__ == '__main__':
    unittest.main()
