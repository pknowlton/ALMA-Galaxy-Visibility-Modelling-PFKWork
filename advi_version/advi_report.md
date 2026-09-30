# ADVI Pipeline Transformation & Architectural Report

**Repository:** `ALMA-Galaxy-Visibility-Modelling-PFKWork`  
**Module:** `advi_version/` (Derived from `dynesty_version_fixed/`)  
**Target Architecture:** Automatic Differentiation Variational Inference (ADVI)  
**Reference Document:** Glenn Moncrieff, *"ADVI vs MCMC"* ([https://gmoncrieff.github.io/posts/advi-vs-mcmc/](https://gmoncrieff.github.io/posts/advi-vs-mcmc/))  
**Scientific Reference:** Kucukelbir, Tran, Ranganath, Gelman, & Blei (2017), *"Automatic Differentiation Variational Inference"*, JMLR 18(14):1–45  
**Models Supported:**
1. `twod_gaussring` (Inclined 2D Gaussian Ring, 7 parameters)
2. `twod_gauss1blob` (Inclined 2D Gaussian Ring + 1 compact Gaussian Blob, 11 parameters)

---

## 1. Executive Summary

This report documents the conversion of the Bayesian interferometric visibility modeling pipeline from dynamic nested sampling (`dynesty_version_fixed/`) into an **Automatic Differentiation Variational Inference (ADVI)** architecture inside `advi_version/`. 

Nested sampling (`dynesty`) and ensemble MCMC (`emcee`) operate by drawing thousands to millions of stochastic samples across the parameter space. While they recover exact posterior probability densities and evidence integrals ($\mathcal{Z}$), they can require dozens to hundreds of CPU-hours on cluster compute nodes. As demonstrated in Glenn Moncrieff's post, **ADVI turns Bayesian inference into an optimization problem**, maximizing the Evidence Lower Bound ($\text{ELBO}$) to approximate posterior probability distributions $\sim 10\times - 100\times$ faster.

In `advi_version/`:
- All physical coordinate systems, Fourier convention fixes, and audit remediation items from `dynesty_version_fixed/` are **strictly preserved**.
- Direct dependencies on **Galario** (GPU Accelerated Library for Analyzing Radio Interferometer Observations) and **CASA** (CASAcore / `casatasks` / `casatools`) are maintained identically without artificial fallbacks.
- The pipeline is fully streamlined to focus on the two requested models (`twod_gaussring` and `twod_gauss1blob`).
- Complete automated unit tests verify coordinate parity, analytical transformation Jacobians, and optimization convergence.

---

## 2. Directory Layout of `advi_version/`

```text
advi_version/
├── advi_report.md                           # This comprehensive architectural report
├── launch_headless_scratch.py              # CANFAR Skaha batch compute session launcher
├── launch_fittings_scratch_bshlog_pfk.sh    # Console & file logging wrapper (tee bashlog.txt)
├── launch_fittings_scratch_psrec_pfk.sh     # System resource profiling (psrecord CPU/RAM)
├── launch_fittings_scratch_pfk.sh           # Core execution script (staging, fitting, imaging)
├── input_dir/
│   ├── run_advi.py                          # Core ADVI optimizer & posterior sampler
│   ├── prior_advi.py                        # Bijective unconstrained parameter transforms & Jacobians
│   ├── model_prof.py                        # 2D Gaussian ring & blob surface brightness models
│   ├── visualize_advi.py                    # ELBO convergence, corner plots, CASA dirty imaging
│   ├── radec_calc.py                        # Sky coordinates, Astropy WCS, & YMC positions
│   ├── ymc_prior_full_err.csv               # Sun et al. (2024) cluster catalog
│   └── test_advi.py                         # Automated test suite for transforms & optimizer
└── stan_reference/                          # Reference implementations matching the blog post
    ├── README.md                            # Mathematical bridge between Stan ADVI and Galario
    ├── ring_visibility_model.stan           # Reference Stan model file
    ├── fit_advi_cmdstan.py                  # Python CmdStanPy invocation script
    └── fit_advi_cmdstan.R                   # R CmdStanR invocation script
```

---

## 3. What Was Transferred Over (Unchanged from `dynesty_version_fixed/`)

The following core physics, conventions, and operational infrastructure were transferred over intact:

1. **UV Visibility Data Ingestion**:
   - `initialize_data(data_file)` in `run_advi.py` preserves full frequency scaling. It automatically detects 6-column tables (`u, v, re, im, weight, freq`) and calculates exact baseline coordinates $u_\lambda = u/\lambda, v_\lambda = v/\lambda$ per datum, preventing single-frequency bandwidth smearing. For pre-scaled 5-column tables, $(u, v)$ are ingested directly as wavelengths.
2. **Descending-RA Coordinate Parity**:
   - `get_grid(nxy, dxy)` in `model_prof.py` maintains Galario's `origin='lower'` geometry:
     - Pixel $[N_{xy}/2, N_{xy}/2]$ is the physical origin $(0, 0)$.
     - RA decreases with column index $ix$ (East is positive, $x > 0$ on the left side of the grid).
     - Dec increases with row index $iy$ (North is positive, $y > 0$ on the top side of the grid).
   - Blob Cartesian positions are strictly $x_{\text{blob}} = r \sin(\text{ang})$ (East) and $y_{\text{blob}} = r \cos(\text{ang})$ (North).
   - Disk Position Angle (PA) is defined East of North across all models without hidden arbitrary offsets.
3. **Audit Parameterization (Log Integrated Flux + Log Size)**:
   - Preserved the transformation from peak surface brightness to **log integrated flux + log size** (`Ring LogFlux`, `Ring LogSigma`, `B1 LogFlux`, `B1 LogSigma`). This completely eliminates the non-linear flux-size "funnel" degeneracy that causes samplers and optimizers to get stuck at near-zero widths.
4. **Physical Constants & Restoring Beam Solid Angle**:
   - `jybm_to_jysr()` retains the mathematically correct Gaussian beam solid angle:
     $$\Omega_{\text{beam}} = \frac{\pi}{4\ln 2} \theta_{\text{maj}} \theta_{\text{min}}$$
     (correcting the factor-of-two bug present in older versions).
5. **CASA Residual Dirty Imaging Pipeline**:
   - `visualize_advi.py` maintains the exact CASA imaging workflow:
     - Subtracts best-fit model visibilities from observed complex visibilities: $V_{\text{resid}} = V_{\text{obs}} - V_{\text{mod}}$.
     - Implants $V_{\text{resid}}$ into Measurement Sets (`.ms`) for both XX and YY polarizations.
     - Runs `tclean` with `niter=0` to create pristine dirty residual maps, preventing non-linear deconvolution artifacts on residual noise.
     - Convolves the intrinsic sky model with the synthesised beam for like-for-like comparison against the restored CLEAN image.
6. **Cluster Scratch Staging Architecture**:
   - `launch_fittings_scratch_pfk.sh` retains the POSIX exit trap (`trap cleanup EXIT`) and staging in local scratch storage (`/scratch`), syncing output logs, plots, FITS images, and checkpoints back to persistent project storage.

---

## 4. What Was Changed

1. **`model_prof.py` (Streamlined Scope)**:
   - dynesty version contained ~1,000 lines supporting 7 different multi-blob configurations (`twod_gauss1blob_2peak`, `twod_gauss1blob_2peak_dp`, `twod_gauss2blob`, `twod_gauss3blob`, `simgauss`).
   - Per user instructions, `advi_version/input_dir/model_prof.py` was trimmed down to include **only the two requested models**:
     - `twod_gaussring` (7 parameters)
     - `twod_gauss1blob` (11 parameters)
   - `model_addon(fittype)` and `model_prof(..., fittype)` now validate against this restricted set and raise explicit, descriptive errors if an unsupported model is invoked.
2. **Prior Architecture (`prior_advi.py` vs. `prior_tform_streamline.py`)**:
   - **Dynesty requirement**: Required an inverse-CDF / quantile transform $F^{-1}(u)$ mapping a unit hypercube $u \in [0, 1]^D$ to physical parameters $\theta$.
   - **ADVI requirement**: Requires bijective, continuously differentiable mappings $T: \text{supp}(p) \to \mathbb{R}^D$ that project constrained physical parameters $\theta$ to **unconstrained real space** $\zeta \in (-\infty, \infty)^D$, along with their inverse $T^{-1}(\zeta)$ and exact transformation log-Jacobian determinants.
3. **Post-Processing & Visualization (`visualize_advi.py` vs. `visualize_dynesty.py`)**:
   - Replaced Dynesty-specific live-point / nested-run progression plots (`dyplot.runplot()`) with **ELBO convergence trajectory plots** that show the stochastic ELBO, rolling average, and convergence status across optimization iterations.
   - Restores variational parameters and draws 5,000 samples directly from the fitted variational distribution for posterior quantiles and corner plots.
   - Retains all 8 diagnostic figures from `visualize_dynesty.py`: ELBO progression, corner plot, 3-panel comparison (Data, Model, Residuals), contour overlay comparison, Sun et al. (2024) cluster overlays, 4-panel multi-scale assessment, 1D horizontal & vertical brightness profile cuts, and the publication-grade parameter summary table.
4. **Execution Scripts**:
   - `launch_fittings_scratch_pfk.sh` now calls `python run_advi.py $4` followed by `python visualize_advi.py $4`.
   - `launch_headless_scratch.py` and `launch_fittings_scratch_bshlog_pfk.sh` point to the `advi_version/` directory tree.

---

## 5. What Was Added Brand New

1. **`run_advi.py` (Core ADVI Fitting Engine)**:
   - Full Automatic Differentiation Variational Inference implementation (Kucukelbir et al. 2017).
   - **Pathwise Gradient Estimator (Reparameterization Trick)**: Evaluates gradients by sampling standard normal noise $\epsilon \sim \mathcal{N}(0, I_D)$ and pushing variations through $\zeta = \mu + \sigma \odot \epsilon$.
   - **Parallel Likelihood Evaluation**: Evaluates $\nabla_\theta \ln \mathcal{L}(\theta) = -\frac{1}{2} \nabla_\theta \chi^2$ via central finite differences parallelized across CPU cores using Python's `multiprocessing.Pool`.
   - **Adam Optimizer**: Updates variational mean $\mu$ and log standard deviation $\omega = \ln \sigma$ using first and second moment tracking with gradient norm clipping to ensure numerical stability during initial exploration.
   - **Stan-Style Convergence Criterion (`tol_rel_obj`)**: Evaluates the relative change in a rolling window of the ELBO:
     $$\frac{|\bar{\text{ELBO}}_{t} - \bar{\text{ELBO}}_{t-W}|}{|\bar{\text{ELBO}}_{t-W}| + \epsilon} < \text{tol\_rel\_obj}$$
     matching the default stopping condition (`tol_rel_obj=0.001`) from Stan's `model.variational()` referenced in the blog post.
   - **Checkpointing**: Exports `./output/<fittype>_advi_checkpoint.npz` containing variational parameters ($\mu, \sigma$), ELBO history, MAP best-fit parameters, and 5,000 posterior draws.
2. **`prior_advi.py` (Analytic Logit/Sigmoid Transforms & Exact Jacobians)**:
   - For bounded interval $\theta_i \in [a_i, b_i]$:
     $$\zeta_i = \operatorname{logit}\left(\frac{\theta_i - a_i}{b_i - a_i}\right), \quad \theta_i = a_i + (b_i - a_i) \cdot \sigma(\zeta_i)$$
   - Disk uniform area radial distance ($p(r) \propto r$ for clump separation $r \in [r_{\min}, r_{\max}]$):
     $$r = \sqrt{r_{\min}^2 + u \cdot (r_{\max}^2 - r_{\min}^2)}, \quad u = \sigma(\zeta)$$
   - The joint unconstrained log-prior and transformation log-Jacobian is:
     $$\ln p(\zeta) = \sum_{i=1}^D \left[ \ln \sigma(\zeta_i) + \ln(1 - \sigma(\zeta_i)) \right]$$
   - The analytical gradient with respect to $\zeta$ is:
     $$\nabla_\zeta \ln p(\zeta) = 1 - 2\sigma(\zeta)$$
     Because this gradient is exact, it introduces zero numerical error.
3. **`test_advi.py` (Automated Test Suite)**:
   - Includes 11 automated unit and integration tests verifying:
     - Numerical stability of $\sigma(z), \operatorname{logit}(u),$ and $\ln(\sigma(z)(1-\sigma(z)))$.
     - Bijective roundtrip transformation consistency ($\theta \to \zeta \to \theta$).
     - Exact match between analytical transformation derivatives $\frac{d\theta}{d\zeta}$ and numerical finite differences.
     - Exact match between analytical prior gradients $\nabla_\zeta \ln p(\zeta)$ and numerical finite differences.
     - Model metadata validity and rejection of unsupported profiles.
     - End-to-end optimization loop execution on synthetic visibility data.
4. **`stan_reference/` (Blog Reference Bridge)**:
   - Provides `ring_visibility_model.stan`, `fit_advi_cmdstan.py`, and `fit_advi_cmdstan.R`.
   - Explains the architectural bridge: why Stan cannot directly trace computational graphs through external C++/CUDA FFT libraries like Galario, and how `run_advi.py` implements the exact ADVI algorithm from the blog post while retaining full Galario compatibility.

---

## 6. How ADVI Works Compared to Dynesty and Emcee

| Metric / Property | `emcee` (Ensemble MCMC) | `dynesty` (Dynamic Nested Sampling) | `advi` (Variational Inference) |
| :--- | :--- | :--- | :--- |
| **Foundational Concept** | **Sampling**: Markov Chain random walk walkers | **Contraction**: Shrinking nested iso-likelihood prior contours | **Optimization**: Maximizing the Evidence Lower Bound ($\text{ELBO}$) |
| **Typical Likelihood Evals** | $200,000 - 1,000,000+$ evaluations | $500,000 - 2,000,000+$ evaluations | $\mathbf{5,000 - 25,000}$ evaluations ($\mathbf{10\times - 50\times}$ speedup) |
| **Evidence $\ln\mathcal{Z}$ Calculation** | Cannot compute directly (requires thermodynamic integration) | **Directly computes $\ln\mathcal{Z}$** and error bars for Bayesian model selection | Approximates $\ln\mathcal{Z}$ from below via the converged $\text{ELBO} \le \ln\mathcal{Z}$ |
| **Posterior Geometry** | Exact asymptotic posterior | Exact asymptotic posterior (weighted samples) | **Gaussian approximation in unconstrained space** ($q(\zeta) = \mathcal{N}(\mu, \operatorname{diag}(\sigma^2))$) |
| **Degeneracy Handling** | Explores arbitrary non-linear bananas & multi-modal distributions | Excels at multi-modal, heavily curved degeneracies | Mean-field assumes independence in unconstrained space; may underestimate variance |
| **Execution Bottleneck** | Walker autocorrelation times & chain mixing | Shrinking prior volume through narrow likelihood peaks | Parallel gradient evaluations $\nabla_\theta \chi^2$ |

### Mathematical Foundation of ADVI

1. **The Objective (ELBO Optimization)**:
   In Bayesian inference, the target posterior is $p(\theta \mid y) = \frac{p(y \mid \theta)p(\theta)}{p(y)}$.
   Instead of slowly wandering through parameter space with MCMC chains, ADVI posits an analytical variational family $q(\zeta \mid \mu, \sigma)$ in unconstrained real space $\zeta \in \mathbb{R}^D$ and minimizes the Kullback-Leibler (KL) divergence to the true posterior:
   $$\operatorname{KL}(q(\zeta) \,||\, p(\zeta \mid y)) = \int q(\zeta) \ln \frac{q(\zeta)}{p(\zeta \mid y)} d\zeta$$
   Because the normalising constant $p(y)$ is independent of $\zeta$, minimizing $\operatorname{KL}(q \,||\, p)$ is mathematically identical to **maximizing the Evidence Lower Bound (ELBO)**:
   $$\operatorname{ELBO}(\mu, \omega) = \mathbb{E}_{q}\left[ \ln p(y \mid \theta(\zeta)) + \ln p(\theta(\zeta)) + \ln |J(\zeta)| \right] + \mathcal{H}(q)$$
   where $\omega = \ln \sigma$, and $\mathcal{H}(q) = \frac{D}{2}(1 + \ln 2\pi) + \sum_{j=1}^D \omega_j$ is the analytical entropy of the variational Gaussian.

2. **The Pathwise Reparameterization Trick**:
   To calculate gradients of the expectation with respect to $\mu$ and $\omega$ without high-variance score-function estimators, ADVI expresses $\zeta$ deterministically using standard normal noise:
   $$\zeta = \mu + \exp(\omega) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I_D)$$
   This enables backpropagation of gradients directly through the expectation:
   $$\nabla_\mu \operatorname{ELBO} = \mathbb{E}_\epsilon \left[ \nabla_\zeta \ln p(y, \zeta) \right]$$
   $$\nabla_\omega \operatorname{ELBO} = \mathbb{E}_\epsilon \left[ \nabla_\zeta \ln p(y, \zeta) \odot (\exp(\omega) \odot \epsilon) \right] + \mathbf{1}$$

3. **Why It Is Fast**:
   Nested sampling and MCMC spend vast amounts of time exploring the diffuse tails of the prior and computing millions of rejected likelihoods. ADVI uses gradient vectors to move directly toward the posterior mode and determine parameter uncertainties within a few hundred iterations. Once converged, drawing 5,000 posterior samples takes under a second by simply drawing $\zeta \sim \mathcal{N}(\mu, \operatorname{diag}(\sigma^2))$ and applying $\theta = T^{-1}(\zeta)$.

4. **Trade-offs to Keep in Mind**:
   As Glenn Moncrieff notes in his article, ADVI is an **approximate** inference technique. In mean-field ADVI (diagonal covariance in unconstrained space), parameter correlations are neglected in the variational family, which can lead to slight underestimation of marginal credible intervals. Comparing ADVI posteriors against existing `dynesty` or `emcee` runs on the same dataset provides an immediate check on whether the approximate posterior intervals are sufficiently conservative for the science goals.

---

## 7. How to Execute the Pipeline

### 7.1 Running Locally (or inside Interactive Terminal / Shell)
```bash
cd advi_version/input_dir

# 1. Fit the 2D Gaussian Ring model (7 parameters)
python run_advi.py twod_gaussring --eta 0.05 --num-iters 1500 --workers 16

# 2. Generate publication PDF plots, CASA residual dirty maps, and parameter summary table
python visualize_advi.py twod_gaussring

# 3. Fit the 2D Gaussian Ring + 1 Blob model (11 parameters)
python run_advi.py twod_gauss1blob --eta 0.05 --num-iters 1500 --workers 16
python visualize_advi.py twod_gauss1blob
```

### 7.2 Running on CANFAR Cloud / Batch Cluster
```bash
cd advi_version

# Launch headless batch container on CANFAR
python launch_headless_scratch.py ngc3351-advi-ring-01 twod_gaussring
python launch_headless_scratch.py ngc3351-advi-blob1-01 twod_gauss1blob
```

### 7.3 Running the Automated Test Suite
```bash
python advi_version/input_dir/test_advi.py
```
Output:
```text
Ran 11 tests in 0.133s
OK
```
