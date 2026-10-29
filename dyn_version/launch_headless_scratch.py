"""
Launch one fit on CANFAR (headless session).

Usage:
    python launch_headless_scratch.py <session_name> <fittype> <data_file>

Example:
    python launch_headless_scratch.py m95-resid resid_gauss3blob M95_C5+C2_cont93_uvtable_freq_residiii.txt

<data_file> is a file name inside data_dir below (or a full path).
The CASA folder used by the visualiser is derived from the data file name
(see launch_fittings_scratch_pfk.sh):
    ..._uvtable_freq.txt          -> casa_dir_freq
    ..._uvtable_freq_residiii.txt -> casa_dir_freq_residiii
"""

import os
import argparse
from canfar.sessions import Session

parser = argparse.ArgumentParser()
parser.add_argument("name", help="Session name (also the output folder name)")
parser.add_argument("fittype", help="Model name, e.g. twod_gauss3blob_nonuni")
parser.add_argument("data_file", help="Visibility table file name (in data_dir) or full path")
pargs = parser.parse_args()

name = pargs.name
fittype = pargs.fittype

in_dir = '/arc/projects/uvdisk_fit/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK/dyn_version/input_dir'
out_dir = '/arc/projects/uvdisk_fit/dynest_product_dir/'
data_dir = '/arc/projects/uvdisk_fit/ngc_3351/JS_data/cont93GHz/'
data_path = os.path.join(data_dir, pargs.data_file)  # a full path given as data_file is used as is

cores = 16
mem = None
out_dir_full = os.path.join(out_dir, name)

image = 'images.canfar.net/skaha/astroml:latest'
cmd = '/arc/projects/uvdisk_fit/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK/dyn_version/launch_fittings_scratch_bshlog_pfk.sh'

arglist = [in_dir, out_dir_full, name, data_path, fittype]
args = ' '.join(arglist)

session = Session()
session_id = session.create(name=name, image=image, cores=cores, ram=mem, kind="headless",
                            cmd=cmd, args=args, env={'sessiontype': 'headless'})
print(session_id)
