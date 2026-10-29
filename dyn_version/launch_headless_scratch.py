r"""
Launch one fit on CANFAR (headless session)
===========================================
The single launcher for every dataset and model.

Usage:
    python launch_headless_scratch.py <session_name> <fittype> <data> [--out-dir DIR] [--dry-run]

    <session_name>  Name of the CANFAR session. Also the name of the output folder.
    <fittype>       Model name (see model_defs.py), e.g. twod_gauss3blob_nonuni
    <data>          Visibility table: a file name inside DEFAULT_DATA_DIR, or a full path.

Examples:
    python launch_headless_scratch.py m95-3blob twod_gauss3blob_nonuni M95_C5+C2_cont93_uvtable_freq.txt
    python launch_headless_scratch.py m95-resid resid_gauss3blob M95_C5+C2_cont93_uvtable_freq_residiii.txt

The CASA folder used for the residual images is chosen from the data file name
(everything after 'uvtable_'):
    M95_C5+C2_cont93_uvtable_freq.txt           ->  casa_dir_freq
    M95_C5+C2_cont93_uvtable_freq_residiii.txt  ->  casa_dir_freq_residiii

Add --dry-run to print what would be launched without starting a session.
"""

import os
import sys
import argparse

# Pick up the shared settings (config.py) and model list (model_defs.py)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'input_dir'))
from config import NCORES, casa_dir_for_data
from model_defs import get_spec

# --- Locations on CANFAR (/arc) ---------------------------------------------
REPO_DIR = '/arc/projects/uvdisk_fit/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK/dyn_version'
IN_DIR = REPO_DIR + '/input_dir'
SESSION_CMD = REPO_DIR + '/launch_fittings_scratch_bshlog_pfk.sh'
DEFAULT_DATA_DIR = '/arc/projects/uvdisk_fit/ngc_3351/JS_data/cont93GHz'
DEFAULT_OUT_DIR = '/arc/projects/uvdisk_fit/dynest_product_dir/'

IMAGE = 'images.canfar.net/skaha/astroml:latest'
RAM = None  # None = CANFAR default


def main():
    parser = argparse.ArgumentParser(description="Launch a dynesty visibility fit as a CANFAR headless session.")
    parser.add_argument("name", help="Session name (also the output sub-folder name).")
    parser.add_argument("fittype", help="Model name, e.g. 'twod_gauss3blob_nonuni'.")
    parser.add_argument("data", help=f"Visibility table file name (looked up in {DEFAULT_DATA_DIR}) or full path.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help=f"Parent output folder (default: {DEFAULT_OUT_DIR}).")
    parser.add_argument("--dry-run", action="store_true", help="Print the session settings and exit without launching.")
    pargs = parser.parse_args()

    # Catch typos now, before a CANFAR session is spent on them
    try:
        get_spec(pargs.fittype)
        data_path = pargs.data if os.path.isabs(pargs.data) else os.path.join(DEFAULT_DATA_DIR, pargs.data)
        casa_dir = casa_dir_for_data(data_path)
    except ValueError as err:
        sys.exit(f"ERROR: {err}")
    if any(c.isspace() for c in pargs.name + pargs.fittype + data_path + pargs.out_dir):
        sys.exit("ERROR: spaces are not allowed in the name, fittype or paths (they are passed on a command line).")

    out_dir_full = os.path.join(pargs.out_dir, pargs.name)
    arglist = [IN_DIR, out_dir_full, pargs.name, data_path, pargs.fittype]
    args = ' '.join(arglist)

    print(f"Session name : {pargs.name}")
    print(f"Model        : {pargs.fittype}")
    print(f"Data table   : {data_path}")
    print(f"CASA folder  : {casa_dir}")
    print(f"Output       : {out_dir_full}")
    print(f"Cores        : {NCORES}")

    if pargs.dry_run:
        print("\n--dry-run: nothing launched.")
        print(f"cmd  = {SESSION_CMD}")
        print(f"args = {args}")
        return

    from canfar.sessions import Session  # imported here so --dry-run works anywhere
    session = Session()
    session_id = session.create(
        name=pargs.name,
        image=IMAGE,
        cores=NCORES,
        ram=RAM,
        kind="headless",
        cmd=SESSION_CMD,
        args=args,
        env={'sessiontype': 'headless'},
    )
    print(f"\nLaunched session: {session_id}")


if __name__ == '__main__':
    main()
