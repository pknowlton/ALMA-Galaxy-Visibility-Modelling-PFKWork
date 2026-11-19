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

The run's priors are read from the prior card prior_cards/<session_name>.pc
(see input_dir/prior_tform.py), which must exist and name <fittype> as its model.
The <model_name>_legacy.pc cards are good starting points.
"""

import os
import sys
import argparse
from canfar.sessions import Session

parser = argparse.ArgumentParser()
parser.add_argument("name", help="Session name (also the output folder and prior card name)")
parser.add_argument("fittype", help="Model name, e.g. twod_gauss3blob_nonuni")
parser.add_argument("data_file", help="Visibility table file name (in data_dir) or full path")
pargs = parser.parse_args()

name = pargs.name
fittype = pargs.fittype

in_dir = '/arc/projects/uvdisk_fit/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK/dyn_version/input_dir'
out_dir = '/arc/projects/uvdisk_fit/results/dyn_results/'
data_dir = '/arc/projects/uvdisk_fit/ngc_3351/JS_data/cont93GHz/'
data_path = os.path.join(data_dir, pargs.data_file)  # a full path given as data_file is used as is
prior_card = os.path.join(os.path.dirname(in_dir), 'prior_cards', name + '.pc')


def card_model(path):
    """Model name from a prior card's 'model:' line (the full check happens in prior_tform.py)."""
    with open(path) as f:
        for line in f:
            text = line.split('#', 1)[0].strip()
            if not text:
                continue
            key, sep, value = text.partition(':')
            if sep and key.strip().lower() == 'model':
                return value.strip()
            break
    return None


if not os.path.isfile(prior_card):
    sys.exit(f"Prior card not found: {prior_card}")
model = card_model(prior_card)
if model != fittype:
    sys.exit(f"Prior card {prior_card} is for model '{model}', but the fittype is '{fittype}'.")

cores = 16
mem = None
out_dir_full = os.path.join(out_dir, name)

image = 'images.canfar.net/skaha/astroml:latest'
cmd = '/arc/projects/uvdisk_fit/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK/dyn_version/launch_fittings_scratch_bshlog_pfk.sh'

arglist = [in_dir, out_dir_full, name, data_path, fittype, prior_card]
args = ' '.join(arglist)

session = Session()
session_id = session.create(name=name, image=image, cores=cores, ram=mem, kind="headless",
                            cmd=cmd, args=args, env={'sessiontype': 'headless'})
print(session_id)
