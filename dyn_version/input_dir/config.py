"""
Pipeline Settings (edit here, once)
===================================
Values that used to be typed in several places live here, so a change is made
in one spot. This module has no imports, so it is safe to import from anywhere
(including the launcher script).
"""

# -----------------------------------------------------------------------------
# Compute
# -----------------------------------------------------------------------------
# CPU cores used by the dynesty multiprocessing Pool (and its queue_size).
# The launcher requests this same number of cores from CANFAR.
NCORES = 16

# -----------------------------------------------------------------------------
# Sky position
# -----------------------------------------------------------------------------
# ALMA observation phase centre of NGC 3351 / M95 (ICRS), as astropy SkyCoord strings.
PHASE_CENTER_RA = '10h43m57.7330s'
PHASE_CENTER_DEC = '+11d42m12.9996s'

# -----------------------------------------------------------------------------
# CASA products
# -----------------------------------------------------------------------------
# The CASA folder is chosen per run from the data table name (see
# launch_fittings_scratch_pfk.sh and visualize_dynesty.py --casa-dir):
#     ..._uvtable_freq.txt          ->  casa_dir_freq
#     ..._uvtable_freq_residiii.txt ->  casa_dir_freq_residiii
# (launch_fittings_scratch_pfk.sh applies the same rule in bash; keep the two in step.)
# These are the file names expected INSIDE every casa_dir_* folder.
CASA_MS_FULL = 'M95_C5+C2_cont93_uvtable.ms'      # combined measurement set (imaged as the observed data)
CASA_MS_XX = 'M95_C5+C2_cont93_trimmedXX.ms'      # XX polarisation measurement set
CASA_MS_YY = 'M95_C5+C2_cont93_trimmedYY.ms'      # YY polarisation measurement set
CASA_DATA_IMAGE = 'M95_cont93GHz_auto'            # base name of the CLEANed data image
CASA_RESID_IMAGE = 'M95_cont93GHz_residual'       # base name of the dirty residual image


def casa_dir_for_data(data_path):
    """
    Returns the CASA folder name that goes with a visibility data table.

    The folder name is 'casa_dir_' + whatever follows 'uvtable_' in the file name:
        M95_C5+C2_cont93_uvtable_freq.txt          ->  casa_dir_freq
        M95_C5+C2_cont93_uvtable_freq_residiii.txt ->  casa_dir_freq_residiii
    """
    base = data_path.replace('\\', '/').split('/')[-1]
    stem = base.rsplit('.', 1)[0] if '.' in base else base
    marker = 'uvtable_'
    if marker not in stem:
        raise ValueError(
            f"Cannot work out the CASA folder from data file name '{base}': "
            f"expected the name to contain '{marker}' (e.g. M95_C5+C2_cont93_uvtable_freq.txt)."
        )
    return 'casa_dir_' + stem.split(marker, 1)[1]
