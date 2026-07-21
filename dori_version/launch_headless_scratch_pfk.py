import os
import numpy as np
import pandas as pd
from canfar.sessions import Session

#set up arglist for cmd
in_dir = '/dori_version/input_dir'
out_dir = '/output_dir/name-of-run'
fittype = 'blob_radex15'

#set the session computing parameters
cores=16
mem=None
name = out_dir.split('/')[-1] if out_dir else "mcmc-run-000"
image='images.canfar.net/skaha/astroml:latest'
cmd = '/dori_version/launch_fittings_scratch_pfk.sh' #this should be the complete path to the .sh script that launches the run_fittings.py script
arglist = [in_dir, out_dir, fittype]
args = ' '.join(arglist)

session = Session()
session_id = session.create(
    name=name,
    image=image,
    cores=cores,
    ram=mem,
    kind="headless",
    cmd=cmd,
    args=args,
    env={'sessiontype':'headless'})
print("Session ID: {}".format(session_id))


#Notes to self:
#monitor the headless session events and logs to look for successful launches.
# https://ws-uv.canfar.net/skaha/v0/session/[id]?view=events
# https://ws-uv.canfar.net/skaha/v0/session/[id]?view=logs