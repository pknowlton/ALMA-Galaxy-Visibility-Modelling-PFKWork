"""
Model Registry: the single place that describes every fittype
==============================================================
Every fittype is built from the same few pieces:

    [ ring block ]  +  [ blob 1 ]  +  [ blob 2 ] ...  (+ optional free geometry)

* Ring block (7 parameters, 8 if non-uniform):
      Ring LogFlux, Ring LogSigma, Ring Rad, Inc, PA, Offset RA, Offset Dec
      (+ Prof. Offset for the non-uniform m=2 ring)
* Each blob (4 parameters):
      LogFlux, LogSigma, Dist, Angle
* Geometry (position angle PA, centroid offsets dRA, dDec) comes from one of:
      'ring'  : the ring block (parameters 4, 5, 6)
      'fixed' : constants from the best-fit 3-blob model (residual-map models)
      'free'  : three extra parameters placed after the blob (simgauss)

`model_prof.py`, `prior_tform.py`, `run_dynesty.py`, `visualize_dynesty.py`
and `radec_calc.py` all read this file, so adding or changing a model means
editing it here once (plus the prior ranges in `prior_tform.py`).

This module has no heavy imports (no numpy, no galario), so it is safe to
import anywhere.
"""

import logging
from dataclasses import dataclass
from typing import Optional

# -----------------------------------------------------------------------------
# Fixed geometry used by the residual-map models (resid_gauss1blob / 3blob).
# These are the best-fit values from the twod_gauss3blob model.
# -----------------------------------------------------------------------------
FIXED_GEOMETRY_POSANGLE_DEG = 18.8479
FIXED_GEOMETRY_DRA_ARCSEC = 0.5973
FIXED_GEOMETRY_DDEC_ARCSEC = 0.5284

# Ring ridge drawn on plots for models that have no ring of their own
# (residual and simgauss models): best-fit 3-blob ring radius and inclination.
FIXED_RING_RAD_ARCSEC = 6.3107
FIXED_RING_INC_DEG = 56.4047

# -----------------------------------------------------------------------------
# Parameter names and units
# -----------------------------------------------------------------------------
RING_LABELS = ["Ring LogFlux", "Ring LogSigma", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec"]
RING_UNITS = ["log(Jy)", "log(arcsec)", "arcsec", "degrees", "degrees", "arcsec", "arcsec"]

NONUNI_LABEL = "Prof. Offset"
NONUNI_UNIT = "degrees"

BLOB_PARAM_NAMES = ["LogFlux", "LogSigma", "Dist", "Angle"]
BLOB_UNITS = ["log(Jy)", "log(arcsec)", "arcsec", "degrees"]

N_RING_PARAMS = 7
N_BLOB_PARAMS = 4
N_GEOMETRY_PARAMS = 3


@dataclass(frozen=True)
class ModelSpec:
    """Structural description of one fittype."""
    name: str
    ring: Optional[str] = None      # None, 'uniform' or 'nonuniform'
    n_blobs: int = 0
    geometry: str = 'ring'          # 'ring', 'fixed' or 'free'

    # ---- structure ---------------------------------------------------------
    @property
    def has_ring(self):
        return self.ring is not None

    @property
    def is_nonuni(self):
        return self.ring == 'nonuniform'

    @property
    def n_ring_params(self):
        if not self.has_ring:
            return 0
        return N_RING_PARAMS + (1 if self.is_nonuni else 0)

    @property
    def blob_start(self):
        """Index of the first blob parameter."""
        return self.n_ring_params

    @property
    def geometry_start(self):
        """Index of the free (PA, dRA, dDec) triplet; only meaningful when geometry == 'free'."""
        return self.blob_start + N_BLOB_PARAMS * self.n_blobs

    @property
    def ndim(self):
        n = self.n_ring_params + N_BLOB_PARAMS * self.n_blobs
        if self.geometry == 'free':
            n += N_GEOMETRY_PARAMS
        return n

    def blob_index(self, k):
        """Index of the first parameter (LogFlux) of blob k (k = 0, 1, ...)."""
        return self.blob_start + N_BLOB_PARAMS * k

    def blob_name(self, k):
        """Display prefix of blob k: 'B1', 'B2', ... (the lone simgauss blob is just 'Blob')."""
        if self.geometry == 'free':
            return "Blob"
        return "B%d" % (k + 1)

    # ---- things the other modules need -------------------------------------
    @property
    def periodic_dims(self):
        """Parameter indices that wrap around (disk PA, and non-uniform ring offset)."""
        dims = []
        if self.has_ring:
            dims.append(4)
        if self.is_nonuni:
            dims.append(7)
        return dims

    @property
    def area_dims(self):
        """Indices of blob radial distances, sampled uniformly in area (not uniformly in radius)."""
        return [self.blob_index(k) + 2 for k in range(self.n_blobs)]

    def labels_units(self):
        """(labels, units) lists in parameter order."""
        if self.geometry == 'free':
            # simgauss: one blob first, then the free geometry
            labels = ["Blob LogFlux", "Blob LogSigma", "Dist", "Angle", "PA", "Offset RA", "Offset Dec"]
            units = ["log(Jy)", "log(arcsec)", "arcsec", "degrees", "degrees", "arcsec", "arcsec"]
            return labels, units
        labels, units = [], []
        if self.has_ring:
            labels += RING_LABELS
            units += RING_UNITS
        if self.is_nonuni:
            labels.append(NONUNI_LABEL)
            units.append(NONUNI_UNIT)
        for k in range(self.n_blobs):
            labels += ["%s %s" % (self.blob_name(k), p) for p in BLOB_PARAM_NAMES]
            units += BLOB_UNITS
        return labels, units

    def geometry_params(self, pars):
        """
        Returns (posangle_deg, dRA_arcsec, dDec_arcsec) for this model.
        Taken from the ring block, the fixed constants, or the free triplet.
        """
        if self.geometry == 'ring':
            return pars[4], pars[5], pars[6]
        if self.geometry == 'fixed':
            return FIXED_GEOMETRY_POSANGLE_DEG, FIXED_GEOMETRY_DRA_ARCSEC, FIXED_GEOMETRY_DDEC_ARCSEC
        i = self.geometry_start
        return pars[i], pars[i + 1], pars[i + 2]

    def display_ring_geometry(self, pars):
        """
        Returns (ring_rad_arcsec, inc_deg, posangle_deg, dRA_arcsec, dDec_arcsec)
        used to draw the dashed ring-ridge ellipse on plots.
        Models without a ring of their own use the fixed best-fit 3-blob values.
        """
        if self.has_ring:
            return pars[2], pars[3], pars[4], pars[5], pars[6]
        return (FIXED_RING_RAD_ARCSEC, FIXED_RING_INC_DEG, FIXED_GEOMETRY_POSANGLE_DEG,
                FIXED_GEOMETRY_DRA_ARCSEC, FIXED_GEOMETRY_DDEC_ARCSEC)


# -----------------------------------------------------------------------------
# The registry. To add a model, add one line here (and its prior ranges in
# prior_tform.py).
# -----------------------------------------------------------------------------
MODEL_SPECS = {spec.name: spec for spec in [
    ModelSpec('twod_gaussring',          ring='uniform'),
    ModelSpec('twod_gaussring_nonuni',   ring='nonuniform'),
    ModelSpec('twod_gauss1blob',         ring='uniform',    n_blobs=1),
    ModelSpec('twod_gauss2blob',         ring='uniform',    n_blobs=2),
    ModelSpec('twod_gauss3blob',         ring='uniform',    n_blobs=3),
    ModelSpec('twod_gauss3blob_nonuni',  ring='nonuniform', n_blobs=3),
    ModelSpec('resid_gauss1blob',        n_blobs=1, geometry='fixed'),
    ModelSpec('resid_gauss3blob',        n_blobs=3, geometry='fixed'),
    ModelSpec('simgauss',                n_blobs=1, geometry='free'),
]}

# Alternate spellings that are accepted on the command line.
ALIASES = {
    'twod_gaussring_nonuni_3blob': 'twod_gauss3blob_nonuni',
    'twod_simgauss': 'simgauss',
}


def get_spec(fittype):
    """Looks up the ModelSpec for a fittype name (aliases allowed)."""
    name = ALIASES.get(fittype, fittype)
    try:
        return MODEL_SPECS[name]
    except KeyError:
        msg = f"Invalid fittype '{fittype}', please choose a valid fitting model, or add a new one into the code."
        logging.warning(msg)
        raise ValueError(msg)
