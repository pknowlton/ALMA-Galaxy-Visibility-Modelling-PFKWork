"""
Reference: Fitting a Stan Model with ADVI in Python using CmdStanPy
====================================================================
Corresponds to the R workflow shown in Glenn Moncrieff's blog post:
https://gmoncrieff.github.io/posts/advi-vs-mcmc/
"""

import numpy as np

def run_cmdstan_advi(data_file='uvtable.txt', seed=123, eta=0.1, tol_rel_obj=0.001):
    try:
        from cmdstanpy import CmdStanModel
    except ImportError:
        print("cmdstanpy is not installed. To install: pip install cmdstanpy && install_cmdstan")
        return

    model = CmdStanModel(stan_file='ring_visibility_model.stan')

    # Load data
    raw = np.loadtxt(data_file, unpack=True)
    u, v, re, im, w = raw[:5]

    data = {
        'N': len(u),
        'u': u,
        'v': v,
        'vis_re': re,
        'vis_im': im,
        'weight': w
    }

    # Execute Automatic Differentiation Variational Inference (ADVI)
    fit_vb = model.variational(
        data=data,
        seed=seed,
        eta=eta,
        tol_rel_obj=tol_rel_obj,
        output_samples=5000
    )

    print("ADVI Variational Sampling Complete!")
    print(fit_vb.variational_sample.head())

if __name__ == '__main__':
    run_cmdstan_advi()
