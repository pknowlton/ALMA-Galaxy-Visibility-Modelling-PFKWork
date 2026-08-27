# ALMA Visibility Modeling: Coordinate System & WCS Alignment Fix

## 1. Executive Summary

During nested sampling analysis of ALMA 93 GHz continuum visibilities for NGC 3351, visualization discrepancies occurred when adding cluster coordinate overlays:
1. In **Figure 2 (Model Contour Overlays)**: The model surface brightness contours were offset from the observed emission clumps in the CLEAN image.
2. In **Figure 3 (Cluster Position Comparisons)**: The prior catalog cluster positions ($\text{Sun et al. 2024}$) and the fitted nested sampling cluster positions ($\text{Dynesty}$) were shifted relative to each other.

This document details the root causes and summarizes the exact corrections applied to `fixing_visualize_dynesty.py` and `radec_calc.py`.

---

## 2. Root Cause Analysis

### Issue A: Missing Centroid Offset in `dynest_radec` (`radec_calc.py`)
* **What was there**:
  ```python
  # ngc3351 = ngc3351_old.spherical_offsets_by(dRA * u.arcsec, dDec * u.arcsec)  # Commented out
  co_ang_b1 = ang_b1 - posangle
  ap_ang_b1 = (360 - co_ang_b1) * u.deg
  radec_b1 = ngc3351.directional_offset_by(position_angle=ap_ang_b1, separation=ap_dist_b1)
  ```
* **Why it caused an offset**:
  - Because `(dRA, dDec)` was commented out, cluster offsets were referenced to the unshifted catalog phase center `ngc3351` rather than the fitted model center `ngc3351.spherical_offsets_by(dRA * u.arcsec, dDec * u.arcsec)`.
* **Fix**:
  - Defined `center = ngc3351.spherical_offsets_by(dRA * u.arcsec, dDec * u.arcsec)` and computed `radec_b` using `center.directional_offset_by(...)`.

### Issue B: WCS Reference Center Double-Shift in `fixing_visualize_dynesty.py`
* **What was there**:
  ```python
  center_mod = SkyCoord((ngc3351.ra.deg - (pars_bf[5]/3600)), (ngc3351.dec.deg - (pars_bf[6]/3600)), unit='deg', frame='icrs')
  mod_wcs = make_model_wcs(center_mod.ra.deg, center_mod.dec.deg, dxy_arcsec, shape=(nxy, nxy))
  ```
* **Why it caused an offset**:
  - `ring_model` is constructed on a grid where $(xx, yy) = (0, 0)$ is the phase center `ngc3351` (since `dra` and `ddec` are already subtracted inside `model_prof`).
  - Passing `center_mod` (`ngc3351 - (dRA, dDec)`) as the WCS reference coordinate `crval` applied the centroid shift a second time.
  - When `Cutout2D` cropped around `ngc3351`, it cropped an offset subregion, and when `ax1.contour(..., transform=ax1.get_transform(cutout_mod.wcs))` was called in Figure 2, the contours were drawn offset from the observed data.
* **Fix**:
  - Constructed `mod_wcs` using the true reference center: `mod_wcs = make_model_wcs(ngc3351.ra.deg, ngc3351.dec.deg, dxy_arcsec, shape=(nxy, nxy))`.

---

## 3. Summary of Applied Fixes

| File | Location | Fix Applied |
| :--- | :--- | :--- |
| **`radec_calc.py`** | `dynest_radec` | Applied `center = ngc3351.spherical_offsets_by(dRA * u.arcsec, dDec * u.arcsec)` so all cluster coordinates include the fitted centroid shift. |
| **`fixing_visualize_dynesty.py`** | `main` | Changed `mod_wcs = make_model_wcs(center_mod...)` to `mod_wcs = make_model_wcs(ngc3351.ra.deg, ngc3351.dec.deg, dxy_arcsec, shape=(nxy, nxy))` to eliminate the double centroid offset. |
| **`model_prof.py`** | Plotting functions | Preserved the original image plane coordinate formulation. |
