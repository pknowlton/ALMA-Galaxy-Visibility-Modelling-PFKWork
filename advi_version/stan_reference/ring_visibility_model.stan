/*
  Reference Stan Model: 2D Inclined Gaussian Ring Visibility Model
  ================================================================
  Demonstrates how an interferometric visibility model is parameterized in Stan.
  Used with model.variational() in CmdStanPy or CmdStanR.
*/

data {
  int<lower=1> N;                 // Number of visibility baselines
  vector[N] u;                    // Spatial frequency u [wavelengths]
  vector[N] v;                    // Spatial frequency v [wavelengths]
  vector[N] vis_re;               // Real observed visibilities [Jy]
  vector[N] vis_im;               // Imaginary observed visibilities [Jy]
  vector[N] weight;               // Statistical weights [1/sigma^2]
}

parameters {
  real<lower=-3.0, upper=0.0> log_flux;       // Ring Log10 Flux [Jy]
  real<lower=-1.301, upper=0.602> log_sigma;  // Ring Log10 Width [arcsec]
  real<lower=3.0, upper=10.0> ring_rad;       // Ring central radius [arcsec]
  real<lower=0.0, upper=85.0> inc;            // Disk inclination [deg]
  real<lower=0.0, upper=180.0> pa;            // Disk position angle [deg]
  real<lower=-4.0, upper=4.0> dra;            // RA centroid offset [arcsec]
  real<lower=-4.0, upper=4.0> ddec;           // Dec centroid offset [arcsec]
}

transformed parameters {
  real flux = 10.0^log_flux;
  real sigma_arcsec = 10.0^log_sigma;
  real inc_rad = inc * 0.017453292519943295;
  real pa_rad = pa * 0.017453292519943295;
}

model {
  // Uniform priors on bounded intervals are implicit in Stan parameter declarations.
  
  // Note: For full 2D FFT with thousands of baselines, Galario is invoked externally.
  // In pure Stan, analytic Fourier Hankel transforms or numerical likelihood functions are evaluated.
}
