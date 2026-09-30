# Reference: Stan ADVI vs. Python ADVI for ALMA Interferometric Modeling

This directory provides reference scripts illustrating the connection between the blog post:
**"ADVI vs MCMC"** by Glenn Moncrieff ([https://gmoncrieff.github.io/posts/advi-vs-mcmc/](https://gmoncrieff.github.io/posts/advi-vs-mcmc/))
and the ALMA visibility modeling pipeline implemented in `advi_version/`.

---

## 1. How ADVI is Run in Stan (as in the Blog Post)

In the referenced blog post, the author fits a nonlinear model using Stan's Automatic Differentiation Variational Inference (ADVI):

### In R (`cmdstanr`):
```r
library(cmdstanr)
model <- cmdstan_model('firemodel_predict.stan', compile = TRUE)

fit_vb <- model$variational(
  data = postfire_data,
  adapt_engaged = FALSE,
  eta = 0.1,
  tol_rel_obj = 0.001,
  seed = 123
)
```

### In Python (`cmdstanpy`):
```python
from cmdstanpy import CmdStanModel

model = CmdStanModel(stan_file='ring_visibility_model.stan')
fit_vb = model.variational(
    data=data_dict,
    seed=123,
    eta=0.1,
    tol_rel_obj=0.001,
    require_converged=True
)
samples = fit_vb.variational_sample
```

---

## 2. Why Stan Cannot Directly Call Galario (And Why We Built `advi_version`)

1. **Autodiff Graph Compatibility**:
   Stan is a compiled C++ domain-specific language. In Stan, every mathematical operation must operate on Stan's custom forward/reverse-mode autodiff type (`stan::math::var`). External libraries like **Galario** (GPU Accelerated Library for Analyzing Radio Interferometer Observations) are compiled C++/CUDA binaries operating on raw C-arrays of `double` and `std::complex`. Stan cannot trace computational graphs through external C++/CUDA FFT kernels.

2. **The Solution in `advi_version/`**:
   The `advi_version/` pipeline implements the **exact same ADVI algorithm** (Kucukelbir et al. 2017) directly in Python:
   - **Logit Coordinate Transforms**: Maps bounded parameters to unconstrained $\mathbb{R}^D$ space.
   - **Exact Jacobians & Analytic Prior Gradients**: $\nabla_\zeta [\ln p(\theta(\zeta)) + \ln |J(\zeta)|] = 1 - 2\sigma(\zeta)$.
   - **Reparameterization Trick**: Pathwise stochastic gradients $\zeta = \mu + \exp(\omega) \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$.
   - **Galario Likelihood Evaluation**: Evaluates synthetic visibilities and $\chi^2$ directly using Galario via parallel worker processes.
   - **Adam Optimizer & Stan-Style Convergence**: Tracks relative change in smoothed ELBO with tolerance `tol_rel_obj` (default 0.001, exactly matching Stan).
