#!/usr/bin/env python3
"""
Minimal 1D Dynesty Dummy Test Script.
Fits a 1D Gaussian target distribution without multiprocessing.
"""

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
    ndim = 1
    sampler = dynesty.NestedSampler(
        log_likelihood, 
        prior_transform, 
        ndim=ndim, 
        nlive=200
    )

    # Run nested sampling
    print("Running sampling...")
    sampler.run_nested(print_progress=True)
    
    # Retrieve and display results
    res = sampler.results
    logz = res.logz[-1]
    
    # Calculate weighted mean parameter
    weights = np.exp(res.logwt - logz)
    mean_val = np.average(res.samples[:, 0], weights=weights)
    
    print("\n" + "="*30)
    print("      RESULTS SUMMARY         ")
    print("="*30)
    print(f"Log Evidence ln(Z) : {logz:.3f}")
    print(f"Recovered Mean     : {mean_val:.3f} (True Target: 3.0)")

if __name__ == "__main__":
    main()
