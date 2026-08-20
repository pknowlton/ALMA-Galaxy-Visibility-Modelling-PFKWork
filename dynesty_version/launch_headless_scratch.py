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

###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%
#In its current form, this script is structured so that for each row of each fit assigned in model_fits, the initial guess parameter is perturbed by a random amount in \pm 5% 
#of the total value range. By changing this initial guess parameter, my hope is to find a better understanding of the errors on these fittings, and the fragility of these convergences.

#In order to launch multiple headless sessions in an organized manner, you must use skaha, which requires python>=3.9. Galario, used for the fitting, requires python<=3.8. 
#Therefore, a series of .py and .sh scripts must be used to go about launching multiple headless sessions in parallel to use in fitting. 

#This version of the tool creates and samples an axisymmetric radial brightness profile. If you want to create and sample 2D images, there is an incomplete 2D version to use instead. 

#launch_headless_sessions.py -> this file dictates the number of fittings to run, the fittypes, the allowed parameter ranges, and interacts with skaha.Session. 
#                               In the session.create call, the file launch_fittings.sh is passed as the command
#launch_fittings.sh ----------> this file allows the fittings to be run, and specifically in an environment where galario (and other fitting dependencies) are installed. 
#                               One of the arguments passed in this file is run_fittings.py
#run_fittings.py -------------> this file actually implements all the model equations, passes them into galario, and runs the MCMC sampling to fit the models to the observed visibilities.
#                               This file utilizes multiprocessing, so that the fits can converge more quickly. In combination with the fact that several fits can be performed 
#                               at once by utilizing the structure in launch_headless_sess.py, this should allow the parameter space to be explored more efficiently. 
#                               One of the arguments passed to this script is global_params.txt
#                               #IMPORTANT aside on multiprocessing: I have found that passing NO arguments to the lnpostfn function in the sampler results in a MAJOR speedup (~10x on my personal computer), but this means using a ton of global variables which is not fantastic. If you need a major speedup, feel free to try it out. 
#global_params.txt -----------> this file contains a handful of parameters global to all fits run, such as the visibilities table and observing frequency. 
#                               It also contains the visualization parameters, which are used in visualizeModel.py
#visualizeModel.py -----------> this file contains functions to visualize the goodness of the model that was converged upon. 

#Again, please note that this should be run in an environment py>=3.9, with skaha, numpy, and pandas installed. 
#In launch_fittings.sh, please provide the path to an environment py<=3.8, with galario, casa, corner, numpy, pandas, emcee, scipy, astropy, uvplot, and matplotlib installed.

#Note that for figure creation, these scripts are currently built such that they will run WITHOUT TeX. This is simply a workaround for the fact that this specific image doesn't 
#have a TeX installation. However, if you are running this WITH TeX installed, then please go uncomment the appropriate lines in run_fittings.py and visualizeModel.py
###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%###%%%

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