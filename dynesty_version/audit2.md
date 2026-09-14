Although each Gaussian profile is mathematically simple, fitting it to complex visibilities is not a simple 1D fitting problem. The most likely difficulties are these, roughly in order of importance.

1. The Gaussian positions are mirrored East_West

The present conversion

xoff = distance * np.sin(angle)

places positive East offsets toward increasing array columns, which Galario interprets as West. The ring is nearly symmetric, so it can still fit well, while off-centre Gaussians fail.

Use:

xoff = -distance * np.sin(angle)
yoff =  distance * np.cos(angle)

The fit must then be rerun; existing posterior angles were inferred under the mirrored geometry.

2. The reported _best_fit_ is not actually a joint best fit

In visualize_dynesty.py, pars_bf is constructed from the separate median of every parameter:

pars_bf[i] = weighted_median(samples[:, i])

A collection of marginal medians need not correspond to any good sample from the posterior. This is especially serious for Gaussian positions because distance, angle, width, and amplitude are strongly correlated.

Circular angles make it worse. For example, samples around 359_ and 1_ describe almost the same position, but their ordinary median can be near 180_.

For residual imaging, use the maximum-likelihood sample:

best_index = np.argmax(res.logl)
pars_bf = res.samples[best_index]

Alternatively, select the sample with the highest posterior weight. Marginal medians remain useful for reporting one-dimensional uncertainties, but not necessarily for constructing a model image.

3. Peak intensity and width are highly degenerate

For a circular 2D Gaussian, the integrated flux is approximately

[
F = 2\pi I_{\rm peak}\sigma^2.
]

If the component is unresolved or marginally resolved, the data constrain its total flux much better than its peak and width separately. A smaller width can therefore be compensated by a much larger peak.

The current fit uses log peak brightness and linear width. A more stable parameterization would be:

total flux
log(width)
x offset
y offset

The peak brightness can then be calculated from flux and width. Also avoid allowing exactly zero width in the prior.

4. Polar coordinates create awkward likelihood geometry

The Gaussian locations are parameterized by distance and angle. In visibility space, these parameters can produce curved, multimodal posterior distributions. They also become poorly behaved near zero distance.

Fit Cartesian offsets instead:

delta_ra_blob
delta_dec_blob

and convert to distance and position angle only for reporting. This also makes the East_West convention easier to test.

5. Multiple Gaussians can exchange identities

When two or three similar Gaussians are fitted simultaneously, the sampler may freely exchange their labels. This produces several equivalent posterior modes and poor marginal summaries.

Possible remedies include:

- Give each component a physically motivated spatial region.
- Order components by position, flux, or width.
- Fit one component first, then add the next.
- Use informative but defensible position priors from the image.

6. The Gaussian model may be too simple for the source

A compact ALMA continuum feature may contain:

- unresolved substructure;
- an elongated or asymmetric shape;
- emission blended with the ring;
- non-Gaussian wings;
- local ring enhancements rather than an independent source.

A circular Gaussian may therefore leave structured residuals even when its total flux is approximately correct. Test an elliptical Gaussian with major axis, axis ratio, and position angle, but only after the parity problem is fixed.

7. The model may not be sufficiently sampled on the image grid

A narrow Gaussian needs several pixels across its width. As a practical check:

Gaussian FWHM = 2.355 _ sigma

Aim for at least three to five pixels across the FWHM. If the fitted width approaches one pixel or less, its Fourier transform will be inaccurate and amplitude_width degeneracy will become severe.

The prior currently permits widths near zero, which can produce effectively point-like or numerically unstable components.

8. The visibility frequency may be approximated too strongly

The code converts all baselines using a fixed frequency:

wavelength = 299792458 / 93e9

If the continuum table contains appreciable bandwidth or multiple spectral windows, each channel should use its own frequency. A single approximate wavelength changes the effective (u,v) coordinates and can bias compact-source positions and widths.

This is less important if the Measurement Set was already averaged to one narrow effective channel, but it should be verified.

9. The statistical weights may be too optimistic

Millions of visibilities do not necessarily represent millions of independent measurements. Calibration errors are correlated across baselines and time. If the weights contain only thermal noise, a small phase or amplitude calibration error can dominate the likelihood and prevent a visually reasonable Gaussian from obtaining a good fit.

Useful checks are:

- Plot normalized real and imaginary residuals.
- Confirm that their scatter is consistent with the weights.
- Bin residuals by baseline length and time.
- Fit a modest additional fractional calibration uncertainty if necessary.
- Compare XX and YY residuals separately.

10. Residual CLEAN can exaggerate the problem

The residual visibilities are currently CLEANed with niter=10000. This can turn phase errors, sidelobes, and noise into apparently compact positive features.

Use niter=0 and inspect the dirty .residual image first. Only consider deconvolution after the visibility residuals themselves are understood.

11. The visual model comparison is not like-for-like

The plotted model is an intrinsic, unconvolved intensity distribution, whereas the CASA data image is convolved with the restoring beam. A narrow model Gaussian can therefore have a much higher peak than the observed image even when the integrated flux is correct.

In addition, the script_s beam-area expression is twice too large:

# Current
np.pi / (2 * np.log(2))

It should be:

# Correct
np.pi / (4 * np.log(2))

The current conversion makes the displayed data brightness too low by a factor of two.

Recommended diagnostic sequence:

1. Fix the East_West Gaussian offset.
2. Fit one Gaussian only.
3. Use Cartesian _RA and _Dec parameters.
4. Parameterize it by integrated flux and log width.
5. Generate synthetic data from known parameters and verify recovery.
6. Use the maximum-likelihood posterior sample for residuals.
7. Inspect residuals directly in visibility space.
8. Make a dirty residual image with niter=0.
9. Add further Gaussians one at a time.
10. Compare beam-convolved model and data images.

The two strongest immediate explanations are the East_West parity error and constructing the displayed model from separate posterior medians. Either can make a valid Gaussian solution appear unsuccessful even if Dynesty has sampled a better solution elsewhere in the posterior.