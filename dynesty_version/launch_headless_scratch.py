"""
Launch Headless Scratch Session for CANFAR Skaha
=================================================
This script launches an asynchronous "headless" batch compute container on the
CANFAR (Canadian Advanced Network for Astronomical Research) cloud infrastructure
using the `canfar.sessions.Session` (Skaha) API.

Overview & Workflow:
1. Parses run name (`name`) and model profile identifier (`fittype`) from command line.
2. Configures compute resources (16 CPU cores, AstroML Docker container).
3. Defines the command payload, pointing to `launch_fittings_scratch_bshlog_pfk.sh`
   with appropriate arguments (`in_dir`, `out_dir_full`, `name`, `data_path`, `fittype`).
4. Creates a remote headless container session on CANFAR and prints the Session ID.

Usage:
    python launch_headless_scratch.py <name> <fittype>

Example:
    python launch_headless_scratch.py ngc3351-gaussring-01 twod_gaussring

Requirements:
    Python >= 3.9 with `canfar` / `skaha`, `numpy`, and `pandas`.
"""

import os
import numpy as np
import pandas as pd
import argparse
from canfar.sessions import Session

# Directory and data paths on the CANFAR /arc filesystem
in_dir = '/arc/home/pknowlton/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK/dynesty_version/input_dir'
out_dir = '/arc/home/pknowlton/dynest_product_dir/'
data_path = '/arc/projects/uvdisk_fit/ngc_3351/JS_data/cont93GHz/M95_C5+C2_cont93_uvtable.txt'

# ------------------------------------------------------------------------------
# Command-Line Argument Parsing
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Launch a headless CANFAR batch container for Dynesty visibility fitting."
)
parser.add_argument(
    "name",
    type=str,
    help="Name of the CANFAR run / session. Also specifies the output subfolder. Use hyphens (-) instead of spaces."
)
parser.add_argument(
    "fittype",
    type=str,
    help="Model profile configuration name as registered in model_prof.py (e.g. 'twod_gaussring')."
)
pargs = parser.parse_args()

name = pargs.name
fittype = pargs.fittype

# ------------------------------------------------------------------------------
# CANFAR Skaha Headless Session Configuration
# ------------------------------------------------------------------------------
cores = 16                                 # Number of CPU cores allocated for multiprocessing pool
mem = None                                 # Default memory allocation (MB)
out_dir_full = os.path.join(out_dir, name) # Target directory where results will be preserved
image = 'images.canfar.net/skaha/astroml:latest' # Docker container image with astronomical tools

# Entrypoint script executed inside the spawned container
cmd = '/arc/home/pknowlton/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK/dynesty_version/launch_fittings_scratch_bshlog_pfk.sh'
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
print("Session ID: {}".format(session_id))


# Notes to self:
# monitor the headless session events and logs to look for successful launches.
# https://ws-uv.canfar.net/skaha/v0/session/[id]?view=events
# https://ws-uv.canfar.net/skaha/v0/session/[id]?view=logs