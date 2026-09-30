r"""
Bayesian Visibility Modeling with Automatic Differentiation Variational Inference (ADVI)
========================================================================================
This script performs Bayesian parameter estimation by fitting 2D brightness profile
models directly to calibrated ALMA interferometric visibility data $(u, v, \text{Re}, \text{Im}, w)$
using **Automatic Differentiation Variational Inference (ADVI)** (Kucukelbir et al. 2017).

Key Concepts & Architecture:
- **Variational Optimization vs. MCMC / Nested Sampling**:
  Instead of drawing millions of sequential samples across parameter space, ADVI converts
  Bayesian posterior inference into an optimization problem. It maximizes the Evidence
  Lower Bound (ELBO):
      $\text{ELBO}(\mu, \omega) = \mathbb{E}_{q(\zeta \mid \mu, \omega)} \left[ \ln p(y, \zeta) \right] + \mathcal{H}(q)$
  where $\mathcal{H}(q) = \frac{D}{2}(1 + \ln(2\pi)) + \sum_{j=1}^D \omega_j$ is the analytical entropy.
- **Unconstrained Parameter Transformation**:
  Physical parameters $\theta \in \text{supp}(p)$ are mapped to unconstrained coordinates $\zeta \in \mathbb{R}^D$
  via smooth bijective transforms (logit/sigmoid) with exact transformation Jacobians.
- **Pathwise Gradient Estimator (Reparameterization Trick)**:
  Stochastic gradients $\nabla_\mu \text{ELBO}$ and $\nabla_\omega \text{ELBO}$ are computed via:
      $\zeta = \mu + \exp(\omega) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I_D)$
- **Adaptive Optimization**:
  Variational parameters are updated via the Adam (Adaptive Moment Estimation) optimizer
  with gradient norm clipping and rolling-window relative ELBO convergence detection
  (`tol_rel_obj`, matching Stan's `model.variational()` stopping criterion).
- **Parallel Gradient Evaluation**:
  Likelihood gradients $\nabla_\theta \ln \mathcal{L}$ are evaluated using parallelized
  finite differences via a multiprocessing worker pool across CPU cores.

Supported Models:
- 'twod_gaussring': 2D Gaussian Ring (7 parameters)
- 'twod_gauss1blob': 2D Gaussian Ring + 1 Gaussian Blob (11 parameters)

Usage:
    python run_advi.py <fittype> [--eta 0.05] [--num-iters 2000] [--num-samples 4] [--tol-rel-obj 0.001] [--draws 5000] [--workers 16] [--seed 42]

Example:
    python run_advi.py twod_gaussring --eta 0.05 --num-iters 2000 --tol-rel-obj 0.001 --workers 16
"""

import os

# 1. Enforce single-threading BEFORE loading C/BLAS extensions to prevent oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import logging
import argparse
import multiprocessing
from multiprocessing import Pool
import numpy as np

from galario.double import get_image_size

from model_prof import model_prof, model_addon
from prior_advi import (
    theta_to_zeta,
    zeta_to_theta,
    grad_zeta_to_theta,
    log_prior_and_jacobian,
    grad_log_prior_and_jacobian,
    get_prior_bounds,
    sample_prior
)

# Global visibility data storage for worker processes
GLOBAL_DATA = None


def initialize_data(data_file):
    r"""
    Loads and prepares interferometric visibility data for Galario.
    Supports files with 6 columns (with frequency) or 5 columns (pre-scaled wavelength).
    """
    raw_data = np.require(np.loadtxt(data_file, unpack=True), requirements='C')

    if len(raw_data) >= 6:
        u_raw, v_raw, re, im, w, freq = raw_data[:6]
        c_light = 299792458.0
        wavelength = c_light / freq
        u = u_raw / wavelength
        v = v_raw / wavelength
    else:
        u, v, re, im, w = raw_data[:5]

    # Calculate optimal image size and pixel scale from baseline extrema
    nx, dx = get_image_size(u, v)

    args = (nx, dx)
    vis_data = (u, v, re, im, w)
    return args, vis_data


def _eval_chi2_worker(task):
    """
    Worker function for parallel chi^2 evaluation across processes.
    """
    pars, args, fittype = task
    global GLOBAL_DATA
    return model_prof(pars, args, GLOBAL_DATA, 'chi2', fittype)


class AdviVisibilityFitter:
    r"""
    Automatic Differentiation Variational Inference (ADVI) optimizer for ALMA visibility models.
    """
    def __init__(self, fittype, args, vis_data, pool=None, eta=0.05, num_samples=4,
                 tol_rel_obj=0.001, seed=42):
        self.fittype = fittype
        self.args = args
        self.vis_data = vis_data
        self.pool = pool
        self.eta = float(eta)
        self.num_samples = int(num_samples)
        self.tol_rel_obj = float(tol_rel_obj)
        self.seed = int(seed)

        np.random.seed(self.seed)

        self.labels, self.units, self.ndim = model_addon(fittype)
        self.bounds = get_prior_bounds(fittype)

        # Variational parameters: mean mu and log standard deviation omega = ln(sigma)
        # Initialize mu at center of unconstrained space (zeta=0 corresponds to median of prior box)
        self.mu = np.zeros(self.ndim, dtype=float)
        # Initialize variational standard deviations to sigma ~= 0.5 (omega = ln(0.5) ~= -0.693)
        self.omega = np.full(self.ndim, -0.693, dtype=float)

        # Adam optimizer state
        self.m_mu = np.zeros(self.ndim, dtype=float)
        self.v_mu = np.zeros(self.ndim, dtype=float)
        self.m_omega = np.zeros(self.ndim, dtype=float)
        self.v_omega = np.zeros(self.ndim, dtype=float)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps_adam = 1e-8

        # Execution tracking metrics
        self.n_evals = 0
        self.elbo_history = []
        self.converged = False

    def _eval_likelihood_and_gradient(self, theta_list):
        r"""
        Computes log-likelihood ln(L) = -0.5 * chi^2 and its gradient d ln(L) / d theta
        for each theta in theta_list using central finite differences.
        """
        N = len(theta_list)
        D = self.ndim
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        span = high - low

        # Step sizes for central finite difference (scaled to parameter span)
        h = np.maximum(1e-4 * span, 1e-6)

        # Build list of evaluation tasks: base points + perturbed points
        tasks = []
        for theta in theta_list:
            tasks.append((theta, self.args, self.fittype))
            for j in range(D):
                tp = theta.copy()
                tm = theta.copy()
                tp[j] += h[j]
                tm[j] -= h[j]
                tasks.append((tp, self.args, self.fittype))
                tasks.append((tm, self.args, self.fittype))

        # Evaluate chi^2 across workers
        if self.pool is not None:
            chi2_results = self.pool.map(_eval_chi2_worker, tasks)
        else:
            chi2_results = [_eval_chi2_worker(t) for t in tasks]

        self.n_evals += len(tasks)

        # Parse results back into log_lik values and gradients
        log_liks = np.empty(N, dtype=float)
        grad_log_liks = np.empty((N, D), dtype=float)

        stride = 1 + 2 * D
        for i in range(N):
            base_idx = i * stride
            base_chi2 = chi2_results[base_idx]
            log_liks[i] = -0.5 * base_chi2

            grad_i = np.empty(D, dtype=float)
            for j in range(D):
                chi2_plus = chi2_results[base_idx + 1 + 2 * j]
                chi2_minus = chi2_results[base_idx + 1 + 2 * j + 1]
                # d chi^2 / d theta_j ~= (chi2_+ - chi2_-) / (2 * h_j)
                dchi2_dtheta = (chi2_plus - chi2_minus) / (2.0 * h[j])
                grad_i[j] = -0.5 * dchi2_dtheta
            grad_log_liks[i] = grad_i

        return log_liks, grad_log_liks

    def compute_elbo_and_gradients(self):
        r"""
        Estimates the ELBO and its gradients with respect to variational parameters (mu, omega)
        using the pathwise gradient estimator (reparameterization trick) over S samples.
        """
        S = self.num_samples
        D = self.ndim
        sigma = np.exp(self.omega)

        # Standard normal noise draws
        eps = np.random.normal(0.0, 1.0, size=(S, D))

        # Variational unconstrained coordinates: zeta = mu + sigma * eps
        zeta_samples = self.mu + sigma * eps

        # Map to physical parameters
        theta_samples = [zeta_to_theta(zeta_samples[s], self.fittype) for s in range(S)]

        # Evaluate likelihood and physical gradient
        log_liks, grad_log_liks = self._eval_likelihood_and_gradient(theta_samples)

        # Accumulate ELBO and gradients
        elbo_lik_prior = 0.0
        grad_mu = np.zeros(D, dtype=float)
        grad_omega = np.zeros(D, dtype=float)

        for s in range(S):
            zeta_s = zeta_samples[s]
            eps_s = eps[s]
            theta_s = theta_samples[s]

            # ln p(zeta) = ln p(theta) + ln |J(zeta)|
            lp_zeta = log_prior_and_jacobian(zeta_s, self.fittype)
            elbo_lik_prior += (log_liks[s] + lp_zeta) / S

            # Gradient of joint log-density in unconstrained space:
            # d ln p(y, zeta) / d zeta = (d ln L / d theta) * (d theta / d zeta) + d ln p(zeta) / d zeta
            dtheta_dzeta = grad_zeta_to_theta(zeta_s, self.fittype)
            grad_lp_zeta = grad_log_prior_and_jacobian(zeta_s, self.fittype)

            grad_joint_zeta = grad_log_liks[s] * dtheta_dzeta + grad_lp_zeta

            # Pathwise gradients w.r.t variational parameters
            grad_mu += grad_joint_zeta / S
            grad_omega += (grad_joint_zeta * (sigma * eps_s)) / S

        # Entropy of multivariate Gaussian variational distribution:
        # H(q) = (D / 2) * (1 + ln(2*pi)) + sum(omega)
        entropy = 0.5 * D * (1.0 + np.log(2.0 * np.pi)) + np.sum(self.omega)
        elbo = elbo_lik_prior + entropy

        # Entropy gradient w.r.t omega: d H(q) / d omega_j = 1.0
        grad_omega += 1.0

        return elbo, grad_mu, grad_omega

    def fit(self, max_iters=2000, checkpoint_file=None):
        r"""
        Executes ADVI optimization using the Adam optimizer with convergence checking.
        """
        logging.info("Starting ADVI optimization loop (max_iters=%d, eta=%.4f, tol_rel_obj=%.5f)...",
                     max_iters, self.eta, self.tol_rel_obj)

        window_size = 40
        rolling_elbo = []

        for step in range(1, max_iters + 1):
            t_start = time.perf_counter()
            elbo, g_mu, g_omega = self.compute_elbo_and_gradients()
            step_time = time.perf_counter() - t_start

            # Gradient norm clipping to safeguard against explosive steps in early exploration
            norm_g = np.sqrt(np.sum(g_mu**2) + np.sum(g_omega**2))
            max_norm = 15.0
            if norm_g > max_norm:
                scale = max_norm / norm_g
                g_mu *= scale
                g_omega *= scale

            # Adam updates for mu
            self.m_mu = self.beta1 * self.m_mu + (1.0 - self.beta1) * g_mu
            self.v_mu = self.beta2 * self.v_mu + (1.0 - self.beta2) * (g_mu**2)
            m_mu_hat = self.m_mu / (1.0 - self.beta1**step)
            v_mu_hat = self.v_mu / (1.0 - self.beta2**step)
            self.mu += self.eta * m_mu_hat / (np.sqrt(v_mu_hat) + self.eps_adam)

            # Adam updates for omega
            self.m_omega = self.beta1 * self.m_omega + (1.0 - self.beta1) * g_omega
            self.v_omega = self.beta2 * self.v_omega + (1.0 - self.beta2) * (g_omega**2)
            m_omega_hat = self.m_omega / (1.0 - self.beta1**step)
            v_omega_hat = self.v_omega / (1.0 - self.beta2**step)
            self.omega += self.eta * m_omega_hat / (np.sqrt(v_omega_hat) + self.eps_adam)

            # Clip omega to avoid variational variance collapse or infinite explosion
            self.omega = np.clip(self.omega, -6.0, 2.0)

            self.elbo_history.append(elbo)
            rolling_elbo.append(elbo)

            # Logging & Progress reporting
            if step % 20 == 0 or step == 1 or step == max_iters:
                mean_elbo = np.mean(rolling_elbo[-min(len(rolling_elbo), 20):])
                logging.info(f"Iter {step:4d}/{max_iters:4d} | ELBO: {elbo:12.3f} (mean: {mean_elbo:12.3f}) | "
                             f"|grad|: {norm_g:8.2f} | Step Time: {step_time:6.2f}s | Total Evals: {self.n_evals}")

            # Convergence Check (Stan tol_rel_obj criterion)
            if step >= window_size * 2:
                w1 = rolling_elbo[-2 * window_size : -window_size]
                w2 = rolling_elbo[-window_size :]
                mean1 = np.mean(w1)
                mean2 = np.mean(w2)
                rel_change = np.abs(mean2 - mean1) / (np.abs(mean1) + 1e-8)

                if rel_change < self.tol_rel_obj:
                    logging.info("ADVI converged! Relative ELBO change (%.6f) < tol_rel_obj (%.6f) at iteration %d",
                                 rel_change, self.tol_rel_obj, step)
                    self.converged = True
                    break

        logging.info("Optimization complete after %d iterations and %d total likelihood evaluations.",
                     len(self.elbo_history), self.n_evals)

    def sample_posterior(self, num_draws=5000):
        r"""
        Draws unconstrained variational samples and transforms them into physical posterior draws.
        """
        sigma = np.exp(self.omega)
        zeta_draws = np.random.normal(self.mu, sigma, size=(num_draws, self.ndim))

        theta_draws = np.empty((num_draws, self.ndim), dtype=float)
        for i in range(num_draws):
            theta_draws[i] = zeta_to_theta(zeta_draws[i], self.fittype)

        # Identify coherent joint MAP best-fit sample
        # Evaluate joint log posterior density for candidate samples
        sub_n = min(num_draws, 500)
        sub_samples = theta_draws[:sub_n]
        sub_zeta = zeta_draws[:sub_n]

        tasks = [(sub_samples[k], self.args, self.fittype) for k in range(sub_n)]
        if self.pool is not None:
            chi2_list = self.pool.map(_eval_chi2_worker, tasks)
        else:
            chi2_list = [_eval_chi2_worker(t) for t in tasks]
        self.n_evals += sub_n

        log_posts = np.empty(sub_n, dtype=float)
        for k in range(sub_n):
            lp = log_prior_and_jacobian(sub_zeta[k], self.fittype)
            log_posts[k] = -0.5 * chi2_list[k] + lp

        best_idx = np.argmax(log_posts)
        best_fit = sub_samples[best_idx].copy()

        return theta_draws, best_fit


def main():
    global GLOBAL_DATA

    parser = argparse.ArgumentParser(
        description="Run Automatic Differentiation Variational Inference (ADVI) for ALMA visibility modeling."
    )
    parser.add_argument("fittype", type=str, choices=['twod_gaussring', 'twod_gauss1blob', 'twod_gauss2blob', 'twod_gauss3blob'],
                        help="Model profile configuration name ('twod_gaussring', 'twod_gauss1blob', 'twod_gauss2blob', 'twod_gauss3blob').")
    parser.add_argument("--eta", type=float, default=0.05,
                        help="Base learning rate / step size for Adam optimizer (default: 0.05).")
    parser.add_argument("--num-iters", type=int, default=1500,
                        help="Maximum number of variational optimization iterations (default: 1500).")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of Monte Carlo samples per ELBO gradient step (default: 4).")
    parser.add_argument("--tol-rel-obj", type=float, default=0.001,
                        help="Relative ELBO convergence tolerance, matching Stan's tol_rel_obj (default: 0.001).")
    parser.add_argument("--draws", type=int, default=5000,
                        help="Number of posterior samples to draw from fitted variational distribution (default: 5000).")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel worker processes for finite-difference evaluation (default: 16).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from existing checkpoint file if available.")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Do not resume from existing checkpoint.")
    pargs = parser.parse_args()

    fittype = pargs.fittype
    eta = pargs.eta
    num_iters = pargs.num_iters
    num_samples = pargs.num_samples
    tol_rel_obj = pargs.tol_rel_obj
    draws = pargs.draws
    workers = pargs.workers
    seed = pargs.seed

    os.makedirs('./output', exist_ok=True)
    logfile = f"./output/{fittype}_advi.log"
    logging.basicConfig(
        filename=logfile,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )

    logging.info("=================================================================")
    logging.info("ADVI Bayesian Visibility Modeling Execution Started")
    logging.info("=================================================================")
    logging.info(f"Fittype: {fittype}")
    logging.info(f"Optimizer Hyperparameters: eta={eta}, max_iters={num_iters}, num_samples={num_samples}, tol_rel_obj={tol_rel_obj}")
    logging.info(f"Workers: {workers}, Random Seed: {seed}")

    labels, units, ndim = model_addon(fittype)
    logging.info(f"Model dimensions: {ndim}")
    logging.info(f"Parameters: {labels}")

    bounds = get_prior_bounds(fittype)
    logging.info("==========Prior Ranges==========")
    for i in range(ndim):
        logging.info(f"{labels[i]:<25} [{bounds[i, 0]:.3f}, {bounds[i, 1]:.3f}] {units[i]}")

    # Load visibility data
    args, GLOBAL_DATA = initialize_data('uvtable.txt')
    check_file = f"./output/{fittype}_advi_checkpoint.npz"

    fit_start = time.perf_counter()

    pool_workers = min(workers, multiprocessing.cpu_count()) if workers > 1 else 1
    logging.info(f"Initializing multiprocessing Pool({pool_workers})...")

    with Pool(pool_workers) as pool:
        fitter = AdviVisibilityFitter(
            fittype=fittype,
            args=args,
            vis_data=GLOBAL_DATA,
            pool=pool,
            eta=eta,
            num_samples=num_samples,
            tol_rel_obj=tol_rel_obj,
            seed=seed
        )

        # Optimization
        fitter.fit(max_iters=num_iters, checkpoint_file=check_file)

        # Posterior generation
        logging.info(f"Drawing {draws} posterior samples from variational distribution...")
        posterior_samples, best_fit = fitter.sample_posterior(num_draws=draws)

    wall_time = time.perf_counter() - fit_start
    logging.info(f"Fitting completed in {wall_time:.2f} seconds ({wall_time / 86400.0:.3f} days)")
    logging.info(f"Total likelihood evaluations: {fitter.n_evals}")
    eval_throughput = fitter.n_evals / max(wall_time, 1e-6)
    logging.info(f"Throughput: {eval_throughput:.2f} evaluations/sec")

    # Save checkpoint
    np.savez_compressed(
        check_file,
        mu=fitter.mu,
        omega=fitter.omega,
        sigmas=np.exp(fitter.omega),
        samples=posterior_samples,
        best_fit=best_fit,
        elbo_history=np.array(fitter.elbo_history),
        labels=np.array(labels),
        units=np.array(units),
        ndim=ndim,
        fittype=fittype,
        wall_time=wall_time,
        n_evals=fitter.n_evals,
        converged=fitter.converged
    )
    logging.info(f"Saved ADVI checkpoint to {check_file}")

    # Report results summary
    logging.info("==========ADVI Parameter Estimation Summary==========")
    logging.info("Coherent Joint Best-Fit (MAP) Parameters:")
    for i in range(ndim):
        logging.info(f"  {labels[i]:<25} (MAP) : {best_fit[i]:.4f} {units[i]}")

    logging.info("Posterior Marginal Quantiles (16th, 50th, 84th):")
    for i in range(ndim):
        q16, q50, q84 = np.percentile(posterior_samples[:, i], [15.865, 50.0, 84.135])
        m_err = q50 - q16
        p_err = q84 - q50
        logging.info(f"  {labels[i]:<25} : {q50:.4f} (+{p_err:.4f} / -{m_err:.4f}) {units[i]}")

    logging.info("ADVI Execution Finished Successfully.")


if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    main()
