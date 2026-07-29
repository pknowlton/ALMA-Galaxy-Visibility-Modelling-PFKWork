#!/usr/bin/env python3
"""
Minimal 1D Dynesty Dummy Test Script.
Fits a 1D Gaussian target distribution without multiprocessing.
"""

import time
import numpy as np
import dynesty

# 1. Define 1D Prior Transform: maps u in [0, 1] to physical bounds [-10, 10]
def prior_transform(u):
    return -10.0 + 20.0 * u

# 2. Define 1D Log-Likelihood: 1D Gaussian centered at 3.0 with std = 1.0
def log_likelihood(theta):
    mu = 3.0
    sigma = 1.0
    return -0.5 * ((theta[0] - mu) / sigma)**2 - np.log(sigma * np.sqrt(2 * np.pi))

# 3. Main execution
def main():
    print("=== Minimal 1D Dynesty Test ===")
    
    # Initialize 1D sampler
    sampler = dynesty.NestedSampler(
        log_likelihood, 
        prior_transform, 
        ndim=1, 
        nlive=200
    )

    # Run nested sampling with timing
    print("Running sampling...")
    t0 = time.perf_counter()
    sampler.run_nested(dlogz=0.1, print_progress=True)
    t1 = time.perf_counter()
    wall_seconds = t1 - t0
    
    # Retrieve results
    res = sampler.results
    
    # Final logz with error
    logz = res.logz[-1]
    logz_err = res.logzerr[-1]
    
    # Best parameter results and 1-sigma error
    weights = np.exp(res.logwt - logz)
    mean = np.average(res.samples, weights=weights, axis=0)
    std = np.sqrt(np.average((res.samples - mean)**2, weights=weights, axis=0))
    
    # Diagnostics
    niter = res.niter
    ncall = np.sum(res.ncall)
    ncall_per_sec = ncall / wall_seconds
    eff = res.eff

    # Compute remaining dlogz at stopping point (before final live points were added)
    logl_live = res.logl[-res.nlive:]
    logvol_stop = res.logvol[res.niter - 1]
    logz_stop = res.logz[res.niter - 1]
    logz_remain = np.max(logl_live) + logvol_stop
    remaining_dlogz = np.logaddexp(logz_stop, logz_remain) - logz_stop

    # Print Results Diagnostics
    print("\n" + "=" * 40)
    print("RESULTS DIAGNOSTICS")
    print("=" * 40)
    print(f"Final Log Evidence ln(Z) : {logz:.3f} +/- {logz_err:.3f}")
    for i in range(len(mean)):
        print(f"Param {i} (Mean +/- 1-sig) : {mean[i]:.3f} +/- {std[i]:.3f}")
    print(f"Current Iteration        : {niter}")
    print(f"Function Calls (ncall)   : {ncall}")
    print(f"ncall / Wall Second      : {ncall_per_sec:.2f}")
    print(f"Efficiency               : {eff:.2f}%")
    print(f"Remaining dlogz          : {remaining_dlogz:.4f}")
    print("=" * 40)

if __name__ == "__main__":
    main()
