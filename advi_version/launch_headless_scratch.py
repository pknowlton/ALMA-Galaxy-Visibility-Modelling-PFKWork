"""
Launch Headless Scratch Session for CANFAR Skaha (ADVI Pipeline)
================================================================
This script launches an asynchronous headless batch compute container on the
CANFAR cloud infrastructure using the `canfar.sessions.Session` (Skaha) API
for the Automatic Differentiation Variational Inference (ADVI) workflow.

Usage:
    python launch_headless_scratch.py <name> <fittype>

Example:
    python launch_headless_scratch.py ngc3351-advi-ring-01 twod_gaussring
    python launch_headless_scratch.py ngc3351-advi-blob1-01 twod_gauss1blob
"""

import os
import argparse
from canfar.sessions import Session

# Directory and data paths on the CANFAR /arc filesystem
in_dir = '/arc/projects/uvdisk_fit/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK/advi_version/input_dir'
out_dir = '/arc/projects/uvdisk_fit/advi_product_dir/'
data_path = '/arc/projects/uvdisk_fit/ngc_3351/JS_data/cont93GHz/M95_C5+C2_cont93_uvtable_freq.txt'

# ------------------------------------------------------------------------------
# Command-Line Argument Parsing
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Launch a headless CANFAR batch container for ADVI visibility fitting."
)
parser.add_argument(
    "name",
    type=str,
    help="Name of the CANFAR run / session. Also specifies the output subfolder. Use hyphens (-) instead of spaces."
)
parser.add_argument(
    "fittype",
    type=str,
    choices=['twod_gaussring', 'twod_gauss1blob'],
    help="Model profile configuration name ('twod_gaussring' or 'twod_gauss1blob')."
)
pargs = parser.parse_args()

name = pargs.name
fittype = pargs.fittype

# ------------------------------------------------------------------------------
# CANFAR Skaha Headless Session Configuration
# ------------------------------------------------------------------------------
cores = 16                                 # CPU cores allocated for multiprocessing pool
mem = None                                 # Default memory allocation (MB)
out_dir_full = os.path.join(out_dir, name) # Target directory where results will be preserved
image = 'images.canfar.net/skaha/astroml:latest' # Docker container image with astronomical tools

# Entrypoint script executed inside the spawned container
cmd = '/arc/projects/uvdisk_fit/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK/advi_version/launch_fittings_scratch_bshlog_pfk.sh'
arglist = [in_dir, out_dir_full, name, data_path, fittype]
args = ' '.join(arglist)

# ------------------------------------------------------------------------------
# Session Creation & Dispatch
# ------------------------------------------------------------------------------
session = Session()
session_id = session.create(
    name=name,
    image=image,
    cores=cores,
    ram=mem,
    kind="headless",
    cmd=cmd,
    args=args,
    env={'sessiontype': 'headless'}
)
print(f"Session ID: {session_id}")
