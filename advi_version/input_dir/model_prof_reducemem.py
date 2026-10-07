r"""
Memory-reduced 2D brightness models for ADVI visibility fitting
===============================================================
Same sky models and coordinate conventions as ``model_prof.py``. Each likelihood
image is one preallocated array: the ring and every blob are added onto that
sheet instead of being stored as separate pictures and summed at the end.

Scratch buffers used for the coordinate and Gaussian arithmetic are reused for
the lifetime of the worker process. ``run_advi.py`` still imports ``model_prof``;
switch that import to this module to use the reduced-memory images.

Conventions, unchanged from ``model_prof.py``:
- RA offset is positive towards East (Galario ``origin='lower'``).
- Dec offset is positive towards North.
- Disk position angle and blob azimuth are East of North.
"""

import logging

import numpy as np
from galario.double import arcsec, chi2Image, deg, sampleImage

_GRID_CACHE = {}
_SCRATCH = {}
_N_SCRATCH = 4


def get_grid(nxy, dxy):
    r"""Coordinate meshgrid matching Galario's ``origin='lower'`` convention."""
    key = (int(nxy), float(dxy))
    if key not in _GRID_CACHE:
        x = -(np.arange(nxy) - nxy / 2.0) * dxy
        y = (np.arange(nxy) - nxy / 2.0) * dxy
        xx, yy = np.meshgrid(x, y)
        _GRID_CACHE[key] = (xx, yy)
    return _GRID_CACHE[key]


def _scratch_buffers(shape):
    r"""Four reusable work arrays of ``shape``. Not safe to share across threads."""
    bufs = _SCRATCH.get(shape)
    if bufs is None:
        bufs = tuple(np.empty(shape, dtype=np.float64) for _ in range(_N_SCRATCH))
        _SCRATCH[shape] = bufs
    return bufs


def _add_gaussring(image, peak, sigma, rad, radius, scale, work):
    r"""Add a Gaussian ring onto ``image``. ``work`` is a scratch array of the same shape."""
    amp = (10.0 ** peak) * (scale * scale)
    np.subtract(radius, rad, out=work)
    np.divide(work, sigma, out=work)
    np.square(work, out=work)
    np.multiply(work, -0.5, out=work)
    np.exp(work, out=work)
    np.multiply(work, amp, out=work)
    np.add(image, work, out=image)


def _add_gaussblob(image, peak, sigma, xx, yy, xoff, yoff, scale, work0, work1):
    r"""Add a circular Gaussian blob onto ``image``."""
    amp = (10.0 ** peak) * (scale * scale)
    np.subtract(xx, xoff, out=work0)
    np.divide(work0, sigma, out=work0)
    np.square(work0, out=work0)
    np.subtract(yy, yoff, out=work1)
    np.divide(work1, sigma, out=work1)
    np.square(work1, out=work1)
    np.add(work0, work1, out=work0)
    np.multiply(work0, -0.5, out=work0)
    np.exp(work0, out=work0)
    np.multiply(work0, amp, out=work0)
    np.add(image, work0, out=image)


def _new_image(shape):
    return np.zeros(shape, dtype=np.float64)


def _compose_chi2_image(xx, yy, inc, peak, sigma, rad, scale, blobs):
    r"""
    One Jy/pixel image. ``blobs`` is a sequence of ``(peak, sigma, xoff, yoff)``.
    """
    image = _new_image(xx.shape)
    s0, s1, s2, s3 = _scratch_buffers(xx.shape)
    np.divide(xx, np.cos(inc), out=s0)
    np.hypot(s0, yy, out=s1)
    _add_gaussring(image, peak, sigma, rad, s1, scale, s2)
    for blob_peak, blob_sigma, xoff, yoff in blobs:
        _add_gaussblob(image, blob_peak, blob_sigma, xx, yy, xoff, yoff, scale, s2, s3)
    return image


def _compose_plot_image(xx, yy, inc, pa, dra, ddec, peak, sigma, rad, blobs):
    r"""
    One Jy/sr image in the observed frame. ``blobs`` use disk-frame offsets,
    placed on the rotated, deprojected coordinates.
    """
    image = _new_image(xx.shape)
    s0, s1, s2, s3 = _scratch_buffers(xx.shape)
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    cos_inc = np.cos(inc)

    np.subtract(xx, dra, out=s0)
    np.subtract(yy, ddec, out=s1)
    np.multiply(s0, cos_pa, out=s2)
    np.multiply(s1, sin_pa, out=s3)
    np.subtract(s2, s3, out=s2)
    np.multiply(s0, sin_pa, out=s3)
    np.multiply(s1, cos_pa, out=s1)
    np.add(s3, s1, out=s0)
    np.divide(s2, cos_inc, out=s1)
    np.hypot(s1, s0, out=s1)

    _add_gaussring(image, peak, sigma, rad, s1, 1.0, s3)
    for blob_peak, blob_sigma, xoff, yoff in blobs:
        _add_gaussblob(image, blob_peak, blob_sigma, s2, s0, xoff, yoff, 1.0, s1, s3)
    return image


def ring_flux_to_peak(log_flux_jy, sigma_rad, rad_rad, inc_rad):
    r"""Log10 ring peak surface brightness from log10 integrated flux."""
    flux_jy = 10.0**log_flux_jy
    cos_inc = max(float(np.cos(inc_rad)), 0.05)
    area_sr = ((2.0 * np.pi)**1.5) * float(rad_rad) * float(sigma_rad) * cos_inc
    i0_jysr = flux_jy / max(area_sr, 1e-30)
    return float(np.log10(max(i0_jysr, 1e-30)))


def blob_flux_to_peak(log_flux_jy, sigma_rad):
    r"""Log10 blob peak surface brightness from log10 integrated flux."""
    flux_jy = 10.0**log_flux_jy
    area_sr = 2.0 * np.pi * (float(sigma_rad)**2)
    i0_jysr = flux_jy / max(area_sr, 1e-30)
    return float(np.log10(max(i0_jysr, 1e-30)))


def _invalid_version(version):
    msg = f"Invalid version '{version}', must be 'chi2', 'vis', or 'plot'"
    logging.warning(msg)
    raise ValueError(msg)


def _blob_offsets(dist, ang):
    return dist * np.sin(ang), dist * np.cos(ang)


###############################################################################
# 1. Model: 2D Gaussian Ring (7 parameters)
###############################################################################

def twod_gaussring_model(peak, sigma, rad, inc, pa, dra, ddec, nxy, dxy, version):
    r"""Inclined, rotated Gaussian ring on one preallocated image."""
    xx, yy = get_grid(nxy, dxy)
    if version in ("chi2", "vis"):
        return _compose_chi2_image(xx, yy, inc, peak, sigma, rad, dxy, ())
    if version == 'plot':
        return _compose_plot_image(xx, yy, inc, pa, dra, ddec, peak, sigma, rad, ())
    _invalid_version(version)


def twod_gaussring(pars, args, vis_data, version):
    r"""
    2D Gaussian ring.
    pars: [Ring LogFlux, Ring LogSigma, Ring Rad, Inc, PA, Offset RA, Offset Dec]
    """
    log_flux, log_sigma, ring_rad, inclination, posangle, dRA, dDec = pars
    nxy, dxy = args
    u, v, re, im, w = vis_data

    inclination *= deg
    posangle *= deg

    sigma_arcsec = 10.0**log_sigma
    sigma_rad = sigma_arcsec * arcsec
    ring_rad_rad = ring_rad * arcsec
    peak_ring = ring_flux_to_peak(log_flux, sigma_rad, ring_rad_rad, inclination)

    if version in ("chi2", "vis"):
        dRA_rad = dRA * arcsec
        dDec_rad = dDec * arcsec
        model_img = twod_gaussring_model(
            peak_ring, sigma_rad, ring_rad_rad, inclination, posangle,
            dRA_rad, dDec_rad, nxy, dxy, version,
        )
        if version == 'chi2':
            return chi2Image(
                model_img, dxy, u, v, re, im, w,
                dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower',
            )
        return np.array(
            sampleImage(
                model_img, dxy, u, v,
                dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower',
            ),
            dtype=np.complex256,
        )
    return twod_gaussring_model(
        peak_ring, sigma_arcsec, ring_rad, inclination, posangle,
        dRA, dDec, nxy, dxy, version,
    )


###############################################################################
# 2. Model: 2D Gaussian Ring + 1 Gaussian Blob (11 parameters)
###############################################################################

def twod_gauss1blob_model(peak, sigma, rad, inc, pa, dra, ddec, peak_b1, sigma_b1, dist_b1, ang_b1, nxy, dxy, version):
    r"""Inclined Gaussian ring plus one blob on one preallocated image."""
    xx, yy = get_grid(nxy, dxy)
    blobs = ((peak_b1, sigma_b1, *_blob_offsets(dist_b1, ang_b1)),)
    if version in ("chi2", "vis"):
        return _compose_chi2_image(xx, yy, inc, peak, sigma, rad, dxy, blobs)
    if version == 'plot':
        return _compose_plot_image(xx, yy, inc, pa, dra, ddec, peak, sigma, rad, blobs)
    _invalid_version(version)


def twod_gauss1blob(pars, args, vis_data, version):
    r"""
    Ring plus one blob.
    pars: [Ring LogFlux, Ring LogSigma, Ring Rad, Inc, PA, Offset RA, Offset Dec,
           B1 LogFlux, B1 LogSigma, B1 Dist, B1 Angle]
    """
    log_flux, log_sigma, ring_rad, inclination, posangle, dRA, dDec, log_flux_b1, log_sigma_b1, dist_b1, ang_b1 = pars
    nxy, dxy = args
    u, v, re, im, w = vis_data

    inclination *= deg
    posangle *= deg
    ang_b1 *= deg

    sigma_arcsec = 10.0**log_sigma
    sigma_b1_arcsec = 10.0**log_sigma_b1
    sigma_rad = sigma_arcsec * arcsec
    sigma_b1_rad = sigma_b1_arcsec * arcsec
    ring_rad_rad = ring_rad * arcsec

    peak_ring = ring_flux_to_peak(log_flux, sigma_rad, ring_rad_rad, inclination)
    peak_b1 = blob_flux_to_peak(log_flux_b1, sigma_b1_rad)

    if version in ("chi2", "vis"):
        dRA_rad = dRA * arcsec
        dDec_rad = dDec * arcsec
        dist_b1_rad = dist_b1 * arcsec
        model_img = twod_gauss1blob_model(
            peak_ring, sigma_rad, ring_rad_rad, inclination, posangle,
            dRA_rad, dDec_rad, peak_b1, sigma_b1_rad, dist_b1_rad,
            ang_b1, nxy, dxy, version,
        )
        if version == 'chi2':
            return chi2Image(
                model_img, dxy, u, v, re, im, w,
                dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower',
            )
        return np.array(
            sampleImage(
                model_img, dxy, u, v,
                dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower',
            ),
            dtype=np.complex256,
        )
    return twod_gauss1blob_model(
        peak_ring, sigma_arcsec, ring_rad, inclination, posangle,
        dRA, dDec, peak_b1, sigma_b1_arcsec, dist_b1,
        ang_b1, nxy, dxy, version,
    )


###############################################################################
# 3. Model: 2D Gaussian Ring + 2 Gaussian Blobs (15 parameters)
###############################################################################

def twod_gauss2blob_model(peak, sigma, rad, inc, pa, dra, ddec, peak_b1, sigma_b1, dist_b1, ang_b1, peak_b2, sigma_b2, dist_b2, ang_b2, nxy, dxy, version):
    r"""Inclined Gaussian ring plus two blobs on one preallocated image."""
    xx, yy = get_grid(nxy, dxy)
    blobs = (
        (peak_b1, sigma_b1, *_blob_offsets(dist_b1, ang_b1)),
        (peak_b2, sigma_b2, *_blob_offsets(dist_b2, ang_b2)),
    )
    if version in ("chi2", "vis"):
        return _compose_chi2_image(xx, yy, inc, peak, sigma, rad, dxy, blobs)
    if version == 'plot':
        return _compose_plot_image(xx, yy, inc, pa, dra, ddec, peak, sigma, rad, blobs)
    _invalid_version(version)


def twod_gauss2blob(pars, args, vis_data, version):
    r"""
    Ring plus two blobs.
    pars: [Ring LogFlux, Ring LogSigma, Ring Rad, Inc, PA, Offset RA, Offset Dec,
           B1 LogFlux, B1 LogSigma, B1 Dist, B1 Angle,
           B2 LogFlux, B2 LogSigma, B2 Dist, B2 Angle]
    """
    log_flux, log_sigma, ring_rad, inclination, posangle, dRA, dDec, log_flux_b1, log_sigma_b1, dist_b1, ang_b1, log_flux_b2, log_sigma_b2, dist_b2, ang_b2 = pars
    nxy, dxy = args
    u, v, re, im, w = vis_data

    inclination *= deg
    posangle *= deg
    ang_b1 *= deg
    ang_b2 *= deg

    sigma_arcsec = 10.0**log_sigma
    sigma_b1_arcsec = 10.0**log_sigma_b1
    sigma_b2_arcsec = 10.0**log_sigma_b2
    sigma_rad = sigma_arcsec * arcsec
    sigma_b1_rad = sigma_b1_arcsec * arcsec
    sigma_b2_rad = sigma_b2_arcsec * arcsec
    ring_rad_rad = ring_rad * arcsec

    peak_ring = ring_flux_to_peak(log_flux, sigma_rad, ring_rad_rad, inclination)
    peak_b1 = blob_flux_to_peak(log_flux_b1, sigma_b1_rad)
    peak_b2 = blob_flux_to_peak(log_flux_b2, sigma_b2_rad)

    if version in ("chi2", "vis"):
        dRA_rad = dRA * arcsec
        dDec_rad = dDec * arcsec
        dist_b1_rad = dist_b1 * arcsec
        dist_b2_rad = dist_b2 * arcsec
        model_img = twod_gauss2blob_model(
            peak_ring, sigma_rad, ring_rad_rad, inclination, posangle,
            dRA_rad, dDec_rad, peak_b1, sigma_b1_rad, dist_b1_rad,
            ang_b1, peak_b2, sigma_b2_rad, dist_b2_rad, ang_b2,
            nxy, dxy, version,
        )
        if version == 'chi2':
            return chi2Image(
                model_img, dxy, u, v, re, im, w,
                dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower',
            )
        return np.array(
            sampleImage(
                model_img, dxy, u, v,
                dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower',
            ),
            dtype=np.complex256,
        )
    return twod_gauss2blob_model(
        peak_ring, sigma_arcsec, ring_rad, inclination, posangle,
        dRA, dDec, peak_b1, sigma_b1_arcsec, dist_b1,
        ang_b1, peak_b2, sigma_b2_arcsec, dist_b2, ang_b2,
        nxy, dxy, version,
    )


###############################################################################
# 4. Model: 2D Gaussian Ring + 3 Gaussian Blobs (19 parameters)
###############################################################################

def twod_gauss3blob_model(peak, sigma, rad, inc, pa, dra, ddec, peak_b1, sigma_b1, dist_b1, ang_b1, peak_b2, sigma_b2, dist_b2, ang_b2, peak_b3, sigma_b3, dist_b3, ang_b3, nxy, dxy, version):
    r"""Inclined Gaussian ring plus three blobs on one preallocated image."""
    xx, yy = get_grid(nxy, dxy)
    blobs = (
        (peak_b1, sigma_b1, *_blob_offsets(dist_b1, ang_b1)),
        (peak_b2, sigma_b2, *_blob_offsets(dist_b2, ang_b2)),
        (peak_b3, sigma_b3, *_blob_offsets(dist_b3, ang_b3)),
    )
    if version in ("chi2", "vis"):
        return _compose_chi2_image(xx, yy, inc, peak, sigma, rad, dxy, blobs)
    if version == 'plot':
        return _compose_plot_image(xx, yy, inc, pa, dra, ddec, peak, sigma, rad, blobs)
    _invalid_version(version)


def twod_gauss3blob(pars, args, vis_data, version):
    r"""
    Ring plus three blobs.
    pars: [Ring LogFlux, Ring LogSigma, Ring Rad, Inc, PA, Offset RA, Offset Dec,
           B1 LogFlux, B1 LogSigma, B1 Dist, B1 Angle,
           B2 LogFlux, B2 LogSigma, B2 Dist, B2 Angle,
           B3 LogFlux, B3 LogSigma, B3 Dist, B3 Angle]
    """
    log_flux, log_sigma, ring_rad, inclination, posangle, dRA, dDec, log_flux_b1, log_sigma_b1, dist_b1, ang_b1, log_flux_b2, log_sigma_b2, dist_b2, ang_b2, log_flux_b3, log_sigma_b3, dist_b3, ang_b3 = pars
    nxy, dxy = args
    u, v, re, im, w = vis_data

    inclination *= deg
    posangle *= deg
    ang_b1 *= deg
    ang_b2 *= deg
    ang_b3 *= deg

    sigma_arcsec = 10.0**log_sigma
    sigma_b1_arcsec = 10.0**log_sigma_b1
    sigma_b2_arcsec = 10.0**log_sigma_b2
    sigma_b3_arcsec = 10.0**log_sigma_b3
    sigma_rad = sigma_arcsec * arcsec
    sigma_b1_rad = sigma_b1_arcsec * arcsec
    sigma_b2_rad = sigma_b2_arcsec * arcsec
    sigma_b3_rad = sigma_b3_arcsec * arcsec
    ring_rad_rad = ring_rad * arcsec

    peak_ring = ring_flux_to_peak(log_flux, sigma_rad, ring_rad_rad, inclination)
    peak_b1 = blob_flux_to_peak(log_flux_b1, sigma_b1_rad)
    peak_b2 = blob_flux_to_peak(log_flux_b2, sigma_b2_rad)
    peak_b3 = blob_flux_to_peak(log_flux_b3, sigma_b3_rad)

    if version in ("chi2", "vis"):
        dRA_rad = dRA * arcsec
        dDec_rad = dDec * arcsec
        dist_b1_rad = dist_b1 * arcsec
        dist_b2_rad = dist_b2 * arcsec
        dist_b3_rad = dist_b3 * arcsec
        model_img = twod_gauss3blob_model(
            peak_ring, sigma_rad, ring_rad_rad, inclination, posangle,
            dRA_rad, dDec_rad, peak_b1, sigma_b1_rad, dist_b1_rad,
            ang_b1, peak_b2, sigma_b2_rad, dist_b2_rad, ang_b2,
            peak_b3, sigma_b3_rad, dist_b3_rad, ang_b3, nxy, dxy, version,
        )
        if version == 'chi2':
            return chi2Image(
                model_img, dxy, u, v, re, im, w,
                dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower',
            )
        return np.array(
            sampleImage(
                model_img, dxy, u, v,
                dRA=dRA_rad, dDec=dDec_rad, PA=posangle, origin='lower',
            ),
            dtype=np.complex256,
        )
    return twod_gauss3blob_model(
        peak_ring, sigma_arcsec, ring_rad, inclination, posangle,
        dRA, dDec, peak_b1, sigma_b1_arcsec, dist_b1,
        ang_b1, peak_b2, sigma_b2_arcsec, dist_b2, ang_b2,
        peak_b3, sigma_b3_arcsec, dist_b3, ang_b3, nxy, dxy, version,
    )


###############################################################################
# Dispatcher & Metadata Functions
###############################################################################

def model_prof(pars, args, vis_data, version, fittype):
    r"""Dispatch to the reduced-memory ring and blob models."""
    if version not in ("chi2", "vis", "plot"):
        _invalid_version(version)

    if fittype == 'twod_gaussring':
        return twod_gaussring(pars, args, vis_data, version)
    if fittype == 'twod_gauss1blob':
        return twod_gauss1blob(pars, args, vis_data, version)
    if fittype == 'twod_gauss2blob':
        return twod_gauss2blob(pars, args, vis_data, version)
    if fittype == 'twod_gauss3blob':
        return twod_gauss3blob(pars, args, vis_data, version)
    msg = (
        f"Invalid or unsupported fittype '{fittype}' in advi_version. "
        "Supported models are 'twod_gaussring', 'twod_gauss1blob', "
        "'twod_gauss2blob', and 'twod_gauss3blob'."
    )
    logging.error(msg)
    raise ValueError(msg)


def model_addon(fittype):
    r"""Parameter labels, units, and dimensionality for ADVI."""
    if fittype == 'twod_gaussring':
        label = ["Ring LogFlux", "Ring LogSigma", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec"]
        unit = ["log(Jy)", "log(arcsec)", "arcsec", "degrees", "degrees", "arcsec", "arcsec"]
        ndim = len(label)

    elif fittype == 'twod_gauss1blob':
        label = [
            "Ring LogFlux", "Ring LogSigma", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec",
            "B1 LogFlux", "B1 LogSigma", "B1 Dist", "B1 Angle"
        ]
        unit = [
            "log(Jy)", "log(arcsec)", "arcsec", "degrees", "degrees", "arcsec", "arcsec",
            "log(Jy)", "log(arcsec)", "arcsec", "degrees"
        ]
        ndim = len(label)

    elif fittype == 'twod_gauss2blob':
        label = [
            "Ring LogFlux", "Ring LogSigma", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec",
            "B1 LogFlux", "B1 LogSigma", "B1 Dist", "B1 Angle",
            "B2 LogFlux", "B2 LogSigma", "B2 Dist", "B2 Angle"
        ]
        unit = [
            "log(Jy)", "log(arcsec)", "arcsec", "degrees", "degrees", "arcsec", "arcsec",
            "log(Jy)", "log(arcsec)", "arcsec", "degrees",
            "log(Jy)", "log(arcsec)", "arcsec", "degrees"
        ]
        ndim = len(label)

    elif fittype == 'twod_gauss3blob':
        label = [
            "Ring LogFlux", "Ring LogSigma", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec",
            "B1 LogFlux", "B1 LogSigma", "B1 Dist", "B1 Angle",
            "B2 LogFlux", "B2 LogSigma", "B2 Dist", "B2 Angle",
            "B3 LogFlux", "B3 LogSigma", "B3 Dist", "B3 Angle"
        ]
        unit = [
            "log(Jy)", "log(arcsec)", "arcsec", "degrees", "degrees", "arcsec", "arcsec",
            "log(Jy)", "log(arcsec)", "arcsec", "degrees",
            "log(Jy)", "log(arcsec)", "arcsec", "degrees",
            "log(Jy)", "log(arcsec)", "arcsec", "degrees"
        ]
        ndim = len(label)

    else:
        msg = f"Invalid or unsupported fittype '{fittype}' in advi_version."
        logging.error(msg)
        raise ValueError(msg)

    return label, unit, ndim
