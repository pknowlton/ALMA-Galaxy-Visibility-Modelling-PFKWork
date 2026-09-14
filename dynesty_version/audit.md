# Audit of ALMA visibility modelling repository

**Repository audited (read-only):** `/arc/projects/uvdisk_fit/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK`  
**Branch/commit observed:** `msc-v2`, `ec7e101`  
**Audit date:** 2026-09-13 UTC  
**Scope:** active Dynesty workflow, archived emcee workflow, bundled Measurement Sets/UV tables, launch scripts, priors, model construction, posterior summarization, and imaging diagnostics.

## Executive verdict

The current numerical results are **not publishable and must be rerun after the P0 corrections below**. The most consequential likelihood-level defect is not the posterior-median plotting choice: it is that all UVW baselines are divided by one wavelength at 93 GHz even though the Measurement Set contains four SPWs centred from 86.35 to 100.12 GHz, with individual channels spanning about 85.64–100.88 GHz. Relative to the true channel UV coordinate, the code's UV scale is about +8.60% at the low-frequency edge and -7.81% at the high-frequency edge; this cannot be repaired by a single rescaling of posterior samples.

The parity concern is real but its consequence needs qualification. The image's horizontal coordinate is reversed relative to Galario's RA convention, so blob angles are left/right mirrored as parameter labels. For the current construction the physical sky angle is approximately `PA - theta`, not `theta`. However, `radec_calc.py` implements the same compensating transformation, and global `dRA`/`dDec` are passed directly to Galario with correct East/North signs. Therefore parity alone does **not** necessarily destroy a likelihood fit if the prior covered the actual source: such a chain may be relabelled. It does invalidate naïve interpretation of `theta`, and narrow informative angle priors may have selected the wrong region.

The coordinate-wise posterior median and factor-of-two beam-area error corrupt post-processing rather than the nested samples. A saved checkpoint would normally remain usable for re-extracting a coherent sample and regenerating plots. No checkpoint, chain, fit log, output FITS image, or diagnostic PDF is present in the repository, and the configured root `/arc/projects/uvdisk_fit/dynest_product_dir/` does not exist on the mounted project tree, so no actual posterior convergence or residual map could be numerically assessed.
# Audit of ALMA visibility modelling repository

**Repository audited (read-only):** `/arc/projects/uvdisk_fit/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK`  
**Branch/commit observed:** `msc-v2`, `ec7e101`  
**Audit date:** 2026-09-13 UTC  
**Scope:** active Dynesty workflow, archived emcee workflow, bundled Measurement Sets/UV tables, launch scripts, priors, model construction, posterior summarization, and imaging diagnostics.

## Executive verdict

The current numerical results are **not publishable and must be rerun after the P0 corrections below**. The most consequential likelihood-level defect is not the posterior-median plotting choice: it is that all UVW baselines are divided by one wavelength at 93 GHz even though the Measurement Set contains four SPWs centred from 86.35 to 100.12 GHz, with individual channels spanning about 85.64–100.88 GHz. Relative to the true channel UV coordinate, the code's UV scale is about +8.60% at the low-frequency edge and -7.81% at the high-frequency edge; this cannot be repaired by a single rescaling of posterior samples.

Sol's parity concern is real but its consequence needs qualification. The image's horizontal coordinate is reversed relative to Galario's RA convention, so blob angles are left/right mirrored as parameter labels. For the current construction the physical sky angle is approximately `PA - theta`, not `theta`. However, `radec_calc.py` implements the same compensating transformation, and global `dRA`/`dDec` are passed directly to Galario with correct East/North signs. Therefore parity alone does **not** necessarily destroy a likelihood fit if the prior covered the actual source: such a chain may be relabelled. It does invalidate naïve interpretation of `theta`, and narrow informative angle priors may have selected the wrong region.

The coordinate-wise posterior median and factor-of-two beam-area error corrupt post-processing rather than the nested samples. A saved checkpoint would normally remain usable for re-extracting a coherent sample and regenerating plots. No checkpoint, chain, fit log, output FITS image, or diagnostic PDF is present in the repository, and the configured root `/arc/projects/uvdisk_fit/dynest_product_dir/` does not exist on the mounted project tree, so no actual posterior convergence or residual map could be numerically assessed.

## 1. Repository reconnaissance

### Active workflow

| Role | File | Relevant locations |
|---|---|---|
| Remote launcher | `dynesty_version/launch_headless_scratch.py` | lines 31–34 define input/output/data paths; 60–83 create the job |
| Scratch orchestration | `dynesty_version/launch_fittings_scratch_{bshlog,psrec,pfk}.sh` | core staging and execution at `launch_fittings_scratch_pfk.sh:68–92` |
| Dynesty driver | `dynesty_version/input_dir/agy_run_dynesty.py` | UV ingest 60–98; likelihood 102–130; sampler 181–221; summaries 253–277 |
| Prior transforms | `dynesty_version/input_dir/prior_tform.py` | ring 24–73; 1 blob 80–134; concentric 2-peak 141–211; double-pendulum 218–292; 3 blob 299–360 |
| Image models/Galario calls | `dynesty_version/input_dir/model_prof.py` | primitives 47–148; grid/Galario paths 159–998; dispatch 1004–1067 |
| Posterior/residual diagnostics | `dynesty_version/input_dir/visualize_dynesty.py` | beam conversion 54–86; posterior summary 162–184; UV/residual calculation 213–250; MS write 252–279; `tclean` 283–337; image comparison 341–519 |
| Coordinate/WCS helper | `dynesty_version/input_dir/radec_calc.py` | parameter-to-sky conversion 24–131; model WCS 133–151 |

Supported active models are the 7-parameter ring, 11-parameter ring+blob, 13-parameter concentric two-Gaussian blob, 15-parameter double-pendulum two-peak model, and 19-parameter ring+three-blob model (`model_prof.py:1047–1060`, `1098–1121`).

### Archived workflow

`emcee_version/input_dir/run_mcmc_fit_scratch_pfk_radex.py` is the historical sampler; `model_profiles_pfk_radex.py` and `model_vis_pfk_radex.py` supply its likelihood/model-visibilities, and `visualize_fit_final_radex.py` supplies diagnostics. It carries forward the same 93 GHz conversion (`run_mcmc_fit_scratch_pfk_radex.py:130–135`), coordinate-wise medians (`visualize_fit_final_radex.py:84–101`), 10,000-iteration residual CLEAN (`195–225`), and wrong beam factor (`28–37`). Its 56 walkers are initialized in a tiny `1e-5` ball about one guess (`run_mcmc_fit_scratch_pfk_radex.py:189–203`), which is especially poor for permutation modes.

`dynesty_version/FIX_package/` is not staged or called by the launch path: the core launcher copies `input_dir`, not `FIX_package` (`launch_fittings_scratch_pfk.sh:68–73`). It is therefore not evidence of an executed fix.

## 2. Verification matrix for Sol's points

The prompt contains ten top-level bullets but calls them eleven points. The matrix splits “best-fit construction” and “circular variables” into separate claims, yielding eleven checks without inventing an omitted claim.

| # | Claim | Code location / direct evidence | Status | Impact |
|---:|---|---|---|---|
| 1 | `xoff = distance*sin(theta)` causes an East–West parity error under Galario conventions. | `model_prof.py:202–204`, `343–345`, `507–509`, `681–683`, `859–861` construct horizontal coordinates increasing with column; blob centers use `+dist*sin(angle)` at `357–358`, `521–522`, `695–700`, `875–883`. Galario requires RA to increase leftward, i.e. decrease with column. | **Confirmed, with qualification** | The angle handedness is reversed. With the subsequent Galario rotation, sky PA is about `PA-theta`. `radec_calc.py:38–42`, `54–58`, `70–82`, `94–114` compensates for this, so the fit is not automatically unusable if its prior covered the source. Parameters and any uncorrected prior interpretation are wrong. |
| 2 | `pars_bf` is assembled from independent marginal medians, not a maximum-likelihood/MAP sample. | `visualize_dynesty.py:176–184` loops over dimensions and assigns each weighted 50th percentile. It never indexes `res.samples[np.argmax(res.logl)]` or another actual joint sample. | **Confirmed** | The vector may lie off the posterior manifold, especially for correlated peak/width/position parameters. Residuals at `244–250` and every model diagnostic use this potentially low-likelihood synthetic point. Nested samples/evidence themselves are unaffected. |
| 3 | Circular/axial variables are summarized and sampled as ordinary linear variables. | Same quantile loop at `176–184`; no circular statistics. Full-circle `Angle 2` prior is `[0,360]` at `prior_tform.py:283–284`; axial ring PA is `[0,180]` at `58–65`; no Dynesty `periodic=` argument at `agy_run_dynesty.py:203–212`. | **Confirmed** | A posterior straddling 0/360 can have a nonsensical median near 180; bounding also treats equivalent endpoints as distant. `[350,390]` is usefully unwrapped, but `[0,360]` is not. Ring PA has 180-degree rather than 360-degree periodicity. |
| 4 | Gaussian widths are linear while peaks are logarithmic, with zero/sub-pixel widths allowed, creating a funnel/flux degeneracy. | `gaussblob_prof` is `10**peak` with linear `sigma` (`model_prof.py:75`, `107`); priors include width zero throughout `prior_tform.py:58–66`, `115–127`, `196`, `263–285`, `333–352`. | **Confirmed** | For a circular blob, integrated flux is proportional to `10**peak * sigma^2`; unresolved data constrain this combination rather than peak and width separately. Near-zero widths are grid-phase dependent and can underflow or produce NaNs at exactly zero. |
| 5 | Component locations are sampled in polar rather than Cartesian coordinates. | Blob priors are `(Dist, Angle)` (`prior_tform.py:123–126`, `197–200`, `275–284`, `341–352`) and transformed with sine/cosine in `model_prof.py`. The ring centre alone is Cartesian `dRA,dDec`. | **Confirmed** | Uniform radius is not uniform area (`p(x,y) proportional to 1/r`), angle is undefined near zero separation, and polar seams complicate sampling. This matters especially for double-pendulum `dist2` including zero. |
| 6 | Multiple Gaussian components have unconstrained permutations/label switching. | In the three-blob prior, blobs 1 and 3 have identical peak/width/distance priors and overlapping angle ranges: `prior_tform.py:341–352`. Blob 2 is angularly separated. Concentric two-peak widths are differentiated (`192`, `196`); double-pendulum roles are hierarchical (`271–284`). Dynesty has no component initialization; historical emcee starts all walkers at one mode (`run_mcmc...py:189–203`). | **Partially True** | Exact/near label exchange is possible for three-blob B1/B3 in the angle overlap, but components are not all unconstrained permutations. The other parameterizations break exchange symmetry, albeit with strong near-coincident degeneracies. |
| 7 | Image grid/pixel sampling is inadequate for narrow components. | `agy_run_dynesty.py:87–96` calls Galario `get_image_size`; actual baseline extrema give `nxy=4096`, `dxy=0.0391567 arcsec`, FOV `160.386 arcsec`. Priors permit `sigma -> 0`; no resolution-aware lower bound or oversampling check exists. | **Partially True** | The grid satisfies Galario's UV-plane Nyquist/FOV heuristic and its centre is correct, but that does not ensure the *model profile* is sampled. Three pixels across FWHM require `sigma >= 0.0499 arcsec`; five require `>=0.0831 arcsec`. Some current priors explicitly enter this unsafe range. Actual posterior widths cannot be checked because outputs are absent. |
| 8 | A scalar 93 GHz wavelength is used despite multi-channel/multi-SPW data. | `agy_run_dynesty.py:87–90`; `visualize_dynesty.py:223–234`; archived emcee equivalents. The input UV table itself says frequency reading failed (external UV table lines 1–3). `M95_C5+C2_cont93_uvtable.listobs.txt:35–40` shows four SPWs centred at 86.3485, 88.2760, 98.3335, and 100.1210 GHz with 8–9 channels; their channel centres span about 85.6376–100.8783 GHz. | **Confirmed** | Relative to the true channel UV coordinate, `u_code/u_true - 1` runs from +8.60% at the low-frequency edge to -7.81% at the high-frequency edge. A size inferred from either edge alone would shift by the reciprocal factors, about -7.92% to +8.47%. Chromatic distortion cannot be repaired by one posterior rescale. This invalidates quantitative fits. |
| 9 | Visibility weights are used as `w=1/sigma^2` in the Gaussian likelihood. | `agy_run_dynesty.py:87`, `95–96` loads column 5; `model_prof.py:271`, `287` (and all other model wrappers) passes it unchanged to `chi2Image`; log likelihood is `-0.5*chi2` at `agy_run_dynesty.py:102–127`. A scan of all 7,258,040 fitted rows found positive finite weights, range 410.315–1666.734. | **Confirmed for ingestion; calibration unverified** | The code-side weighting algebra matches Galario's complex-visibility chi-square. The repository lacks the UV-table exporter/provenance checks, weight-spectrum validation, weight rescaling, or a calibration/jitter term. Statistical errors may therefore be miscalibrated even though weights are multiplied correctly. |
| 10 | Residual imaging applies `niter=10000`, risking CLEANing noise. | `visualize_dynesty.py:306–324` runs Hogbom `tclean` on residual MSs with `niter=10000`; data imaging also uses 10,000 at `286–304`. | **Partially True** | The configured ceiling is confirmed, but “excessive iterations actually executed” cannot be proven without logs: fixed thresholds may stop earlier. A residual diagnostic should first be a dirty image (`niter=0`) under identical gridding/weighting; any residual deconvolution needs justified thresholds/masks and reported iteration counts. |
| 11 | Beam area uses `pi/(2 ln 2)` instead of `pi/(4 ln 2)`. | Docstring gives the correct expression at `visualize_dynesty.py:54–64`, but executable line 83 uses `np.pi/(2*np.log(2))`. Historical emcee uses the same at `visualize_fit_final_radex.py:35`. | **Confirmed** | Beam area is twice too large, so dividing Jy/beam by it makes data and residual surface brightness exactly a factor of two too low. Model Jy/sr is not reduced by this factor, making the comparison internally inconsistent. |

## 3. Independent checks and additional findings

### P0-level additional defects

1. **Frequency information was discarded at extraction.** The fit table header explicitly reads `wavelength[m] = Frequency reading failed for this MS.` It contains only `u, v, Re, Im, weight`; with millions of channel samples but no frequency/SPW/channel field, correct per-row wavelength conversion is impossible downstream. Re-extract from the MS with either UV coordinates already in wavelengths for each channel or an explicit frequency column. The model may also need a spectral-index term across this ~15% fractional bandwidth.

2. **A hidden fixed 15-degree rotation changes the meaning of PA between models.** `model_prof.py:150–153` defines `PA_RAD=15 deg`. All blob-bearing ring models apply it internally (e.g. `350–354`, `514–518`, `688–692`, `866–870`) and then pass the sampled `posangle` to Galario as another rotation (e.g. `444`, `609`, `801`, `986`). The ring-only model does not apply the fixed angle (`206–212`, `287`). Consequently the reported PA is not a common physical parameter across model classes, and log-evidence comparisons do not compare consistently parameterized models.

3. **Shared-parameter priors differ materially between model classes.** Ring-only uses peak/width/radius/inclination/PA ranges `[-5,15]`, `[0,10]`, `[0,20]`, `[0,90]`, `[0,180]` (`prior_tform.py:58–65`), while two-peak models use `[0,8]`, `[0,3]`, `[4,8]`, `[45,80]`, `[-10,10]` (`182–188`, `263–270`). Other blob models use still other shared priors. Evidence is intrinsically prior-dependent; Bayes factors advertised in the README are not interpretable as component-count comparisons unless shared physical parameters and angle conventions have the same justified priors.

4. **The active visualization is guaranteed to raise `NameError`.** `visualize_dynesty.py:561–562` comments out definitions of `cy_clamp` and `cx_clamp`, but lines `590–593` use both. `dynest_radec` returns at least one coordinate even for ring-only (`radec_calc.py:118–122`), so every normal model enters this code. PDF closure at line 636 and the final completion log are never reached. An older file in `FIX_package` defines the variables, but it is not executed.

### Coordinate, origin, and WCS findings

5. **Galario centre is correct; RA handedness is not.** Algebraically, the active grid expression is `(arange(nxy)-nxy/2)*dxy`, placing zero at array index `nxy/2`, exactly Galario's convention for even images. There is no `N/2` versus `N/2+1` bug in the visibility model. The problem is that Galario's RA coordinate must descend with column; using Galario's `get_coords_meshgrid(..., origin='lower')` would avoid the manual sign error. The Dec array increases with row and is correct for `origin='lower'`.

6. **Phase centre matches the MS.** Current hard-coded phase centre `10:43:57.7330 +11:42:12.9996` (`visualize_dynesty.py:108`, `366`; `radec_calc.py:28`) agrees with both fields in `listobs` to substantially below a milliarcsecond in Dec and about 0.0006 arcsec in RA (`listobs:31–34`). There is no evidence here for a phase-centre mismatch. The repository does not document the continuum-subtraction/concatenation phase handling, so that upstream step remains unaudited.

7. **Ring-only coordinate annotation ignores fitted centre offsets.** `radec_calc.py:118–122` returns the phase centre for `twod_gaussring`, even when fitted `dRA,dDec` are nonzero. This affects annotation/cutout placement, not the likelihood.

8. **Plot WCS and model size are brittle.** `visualize_dynesty.py:354–368` sets model size to `data_plot.shape[0] + 1` and invents a separate WCS. For the present pixel scale this appears intended to turn a 919-pixel cutout into an even 920-pixel model, but the invariant is undocumented and not asserted. Use a shared explicit grid/WCS and verify round trips for centre and injected East/North point sources.

### Sampler and posterior findings

9. **The “dynamic” run disables dynamic batches.** `agy_run_dynesty.py:203–221` creates `DynamicNestedSampler(nlive=250, bound='multi', sample='rwalk')` but calls `run_nested(..., maxbatch=0)`. This permits only the baseline batch, defeating the advertised adaptive allocation. Dynesty documents `maxbatch` as the maximum number of added batches.

10. **`nlive=250` and `dlogz_init=0.5` are weak defaults for the 15–19D multimodal cases.** They are not automatically invalid, and `multi`/`rwalk` are defensible choices, but mode die-off and inaccurate evidence must be tested using multiple independent seeds, larger live-point counts, tighter stopping, and a `slice`/`rslice` comparison. No random seed is recorded, so runs are not reproducible. There is no effective posterior sample target.

11. **Checkpointing is not resumption.** A checkpoint filename is supplied (`agy_run_dynesty.py:197–220`), but every invocation constructs a new sampler; it never restores or passes `resume=True`. Interrupted expensive runs are therefore not reliably resumed.

12. **Absolute log evidence lacks the Gaussian normalization.** The likelihood keeps only `-chi2/2`. This is acceptable for Bayes factors among models using exactly the same data and fixed weights because the omitted term is constant, but the printed absolute `ln Z` is not a normalized data probability, and comparisons across changed selections/weights are invalid.

13. **Likelihood allocations are unnecessarily extreme.** Every call rebuilds multiple 4096x4096 coordinate/intermediate arrays; 16 forked workers can demand many tens of GB. Precompute immutable grids and fixed transforms per worker, or use analytic Gaussian Fourier transforms. This is mainly stability/performance but can cause job termination without a scientific diagnostic.

### Residual and unit findings

14. **Intrinsic model and restored CLEAN image are compared directly.** `visualize_dynesty.py:363–375` creates an intrinsic Jy/sr sky model; it is displayed beside the restored data at `379–392` without convolution by the restoring beam. `imsmooth` is imported but unused. Peaks and compact widths therefore cannot be compared visually or via image-plane profiles. Convolve the model with the exact data restoring beam (and apply the same pixel/WCS response), or form all comparisons as forward-modelled dirty/restored products.

15. **Residual cleaning and PB correction hide the most useful diagnostic.** The code should retain the un-deconvolved, un-PB-corrected residual dirty image and PSF for noise statistics. PB correction amplifies noise away from the centre. CASA documents `niter=0` as the dirty/initial residual mode. If a CLEANed residual is also made, keep it secondary and record the stopping reason.

16. **Hard-coded CLEAN mask and thresholds are dataset-specific.** The mask centre `[2048,2052]` and 400-pixel radius (`visualize_dynesty.py:303`, `323`) assume the inferred grid and cover only about 15.66 arcsec radius, while the model priors permit a ring radius of 20 arcsec plus a 10 arcsec width. Thresholds are fixed literals rather than measured multiples of dirty-image RMS.

17. **Residual-MS implantation relies on undocumented flattening order.** `visualize_dynesty.py:257–277` reshapes a flat UV-table vector into the CASA `DATA` column with no check of row/channel mapping. The bundled XX and YY UV tables do have identical UV coordinates in all 7,258,040 rows, and the same model can be subtracted from both polarizations. That does not prove that flat-table order matches CASA column order. Re-extraction should preserve row/channel identifiers and include an exact round-trip assertion before writing.

18. **No weight/calibration diagnostics are produced.** All fitted weights are finite and positive. The first inspected rows show that the combined visibility is the inverse-variance weighted XX/YY mean and its weight is the sum of polarization weights, which is correct. Still required are reduced chi-square by SPW/time/baseline, standardized residual distributions, amplitude-vs-UV-distance and phase residuals, and an optional fractional amplitude calibration nuisance parameter.

19. **Profile nomenclature is ambiguous.** “Width” is actually Gaussian `sigma`, not FWHM (`model_prof.py:52`, `82`). Every table/plot should state sigma explicitly or convert to FWHM using `2.35482*sigma`.

## 4. Assessment of existing results

### What could be salvaged from a checkpoint

- **Coordinate-wise median bug:** fully salvageable by reloading the checkpoint and selecting an actual joint sample. It never changed the likelihood exploration.
- **Beam-area factor, residual CLEAN choice, missing beam convolution, and plotting `NameError`:** salvageable by rerunning post-processing only.
- **Angle handedness:** often salvageable by relabelling angles and regenerating coordinates, conditional on the intended source lying within the *actual mirrored prior support*. Inspect posterior/prior support before accepting this.
- **Hidden 15-degree PA:** potentially salvageable as a deterministic PA reinterpretation for a fixed model class, but not for evidence comparisons across ring-only versus blob-bearing classes with inconsistent priors.

### What requires a new fit

- The per-channel frequency/UV conversion error changes the likelihood differently for every channel and SPW. All quantitative sizes, offsets, compact-component widths/fluxes, residuals, chi-square values, and evidences require refitting from a frequency-aware visibility table.
- Any run whose posterior admits `sigma < 0.05 arcsec` on the present grid must be repeated with an analytic or sufficiently oversampled model and a resolution-aware prior.
- Model-comparison evidences must be recomputed with consistent shared-parameter priors, coordinate conventions, and validated nested-sampling settings.

Because no actual output artifact was available, it is impossible to determine whether a particular run converged, hit a prior boundary, suffered label exchange, or sampled sub-pixel widths. Existing numbers should be treated as exploratory only.

## 5. Ranked action plan for the student

### P0 — blockers / critical correctness

- [ ] **Rebuild the visibility export with frequency retained per datum.** Prefer output columns `(u_lambda, v_lambda, Re, Im, weight, frequency, spw, channel, row-id)` after flags are applied. Never infer all wavelengths from the filename. Assert a few exported rows directly against MS `UVW`, `CHAN_FREQ`, `DATA_DESC_ID`, channel index, `DATA`, `FLAG`, and `WEIGHT_SPECTRUM`/`WEIGHT`.
- [ ] **Refit every scientific model using per-channel `u,v` in wavelengths.** If one achromatic continuum model is retained, justify it; otherwise add a reference-frequency intensity plus spectral index. Compute the grid conservatively from the largest true UV radius.
- [ ] **Replace hand-built coordinates with Galario's coordinate helper or exactly reproduce its descending-RA, ascending-Dec lower-origin grid.** Define one public convention: `dRA>0` East, `dDec>0` North, PA East of North. Then use `x_blob = r*sin(PA)` on the correctly handed RA coordinate. Remove compensating sign algebra from `radec_calc`.
- [ ] **Add deterministic visibility-domain parity tests before refitting.** Inject a single point/Gaussian at +East, -East, +North, -North; compare synthetic visibility phases with the analytic shift theorem and recover the known offsets. Add a nonzero-PA asymmetric source test and a WCS round-trip test.
- [ ] **Remove the hidden `PA_RAD=15 deg` transform.** If 15 degrees is prior knowledge, express it as the same physical PA prior in every model, not a second internal rotation. Ensure ring-only is nested consistently inside every blob model.
- [ ] **Make shared priors identical across models used for evidence comparison.** Document prior scientific rationale and units; rerun evidences. Do not compare old logZ values across the current inconsistent prior volumes.
- [ ] **Repair guaranteed post-processing failure** by defining/clamping `cx_clamp,cy_clamp` and adding a smoke test that completes the PDF/FITS pipeline for every fittype.

### P1 — sampling and numerical stability

- [ ] **Reparameterize each Gaussian as log integrated flux plus log size** (or log FWHM), with a strictly positive, resolution-aware lower size bound. For blobs, `F = 2*pi*I0*sigma^2`; derive the corresponding ring flux parameterization and include inclination consistently.
- [ ] **Use Cartesian East/North offsets for freely located components**, or if polar priors are scientifically desired, use the correct Jacobian (`r^2` uniform for uniform area) and an explicit circular parameter. Never include `r=0` while retaining a meaningful angle.
- [ ] **Break label symmetry deliberately.** Assign non-overlapping spatial regions tied to named sources, or impose a deterministic ordering (with careful treatment at the angular seam). Verify that relabelling leaves the physical posterior unchanged.
- [ ] **Represent circular/axial variables correctly.** Mark truly periodic Dynesty dimensions where supported, keep posterior intervals unwrapped around a chosen mode, and report circular means/credible arcs. Treat disk PA as 180-degree axial data, not a 360-degree direction.
- [ ] **Make the dynamic sampler genuinely dynamic:** remove `maxbatch=0`; set and justify `nlive_init`, `nlive_batch`, a posterior effective-sample target, and tighter evidence tolerance. For 15–19D models, benchmark at least `multi+rwalk` against `multi+slice/rslice`.
- [ ] **Run at least 3 independent recorded seeds** at increasing live-point counts. Require stable posterior summaries and logZ within estimated uncertainties; inspect mode occupancy, effective sample size, call efficiency, bound diagnostics, and prior-edge pile-up.
- [ ] **Implement tested checkpoint resume** and record package versions, seed, input-table checksum, model/prior version, command line, and complete configuration in each output directory.
- [ ] **Select coherent posterior representatives.** For “best fit,” use `res.samples[argmax(res.logl)]` (MAP equals ML only for constant priors in physical coordinates); separately retain the highest quadrature-weight sample if desired and draw multiple posterior predictive samples. Never build a synthetic point from marginal medians.
- [ ] **Precompute model grids/fixed transforms** and reduce per-likelihood memory. Validate any optimization against analytic Gaussian visibilities and the original calculation at well-resolved widths.

### P2 — diagnostics and verification

- [ ] **Generate visibility diagnostics first:** real/imaginary residual vs `u`, `v`, UV distance, frequency/SPW, time, baseline, and polarization; binned complex residuals; standardized residual histogram/Q-Q plot; total and per-subset chi-square.
- [ ] **Make the primary residual image dirty (`tclean niter=0`)** with the same weighting, cell, imsize, phase centre, and data selection as the observed image. Preserve PSF, sum-of-weights, un-PB-corrected residual, and RMS region. Make any CLEANed residual an explicitly secondary product.
- [ ] **Correct beam solid angle** to `Omega_beam = pi/(4 ln 2) * BMAJ * BMIN` in radians squared. Add a unit test using a known Gaussian beam and label FITS `BUNIT` consistently.
- [ ] **Convolve the intrinsic sky model with the exact restoring beam** before image-plane comparisons and 1D profiles. Also provide a model dirty image made through the observed sampling/weighting for apples-to-apples residual checks.
- [ ] **Derive masks and thresholds from WCS/source extent and measured RMS**, not hard-coded pixel locations or absolute values. Record `iterdone`, stopping reason, peak residual, beam, and RMS in the report.
- [ ] **Keep PB-corrected and uncorrected products distinct.** Use uncorrected maps for stationary-noise residual tests; use PB-corrected maps only for surface-brightness presentation within a declared PB cutoff.
- [ ] **Verify MS implantation ordering** with row/channel identifiers and a write-read round trip on a disposable MS outside the repository. Assert XX/YY UV coordinate equality before sharing model visibilities.
- [ ] **Add end-to-end synthetic recovery tests** spanning multiple SPWs, known East/North offsets, a 0/360 angle seam, near-resolution widths, and two exchangeable blobs. Recovery must be correct in both visibilities and WCS plots.

## 6. Quantitative audit notes

- Fitted table rows: 7,258,040 complex samples.
- Projected baseline range: 12.4371–2616.9985 m.
- At the code's assumed 93 GHz: UV radius 3.858–811.831 klambda.
- Galario default grid inferred from those UV coordinates: `4096 x 4096`, `0.0391567 arcsec/pixel`, `160.386 arcsec` FOV; approximate MRS 32.077 arcsec under Galario's heuristic.
- Minimum Gaussian sigma for 3 pixels/FWHM: 0.0499 arcsec; for 5 pixels/FWHM: 0.0831 arcsec.
- Combined-table weights: 410.315–1666.734, all finite/positive. XX weights: 193.230–910.194; YY: 154.867–892.457. All 7,258,040 XX/YY UV coordinates agree row by row.
- SPW centre frequencies: 86.3485, 88.2760, 98.3335, 100.1210 GHz; channel-centre extrema are approximately 85.6376 and 100.8783 GHz. At the SPW centres, `u_true/u_code = frequency/93 GHz` gives 0.9285, 0.9492, 1.0573, and 1.0766 respectively.

## 7. Reference conventions used in this audit

- [Galario image/coordinate specifications](https://mtazzari.github.io/galario/tech-specs.html): RA increases leftward, Dec upward for `origin='lower'`, and the physical origin is pixel `[Nxy/2,Nxy/2]`.
- [Galario Python API](https://mtazzari.github.io/galario/py-api.html): image units are Jy/pixel; `dxy` is radians; UV coordinates are wavelengths; positive `dRA` and `dDec` translate East and North; Galario's chi-square weights both real and imaginary residual terms.
- [Galario coordinate-grid cookbook](https://mtazzari.github.io/galario/cookbook.html): authoritative descending-RA grid construction.
- [Dynesty dynamic sampling documentation](https://dynesty.readthedocs.io/en/v2.1.5/dynamic.html): baseline plus added batches and the meaning of `maxbatch`.
- [CASA `tclean` documentation](https://casadocs.readthedocs.io/en/latest/api/tt/casatasks.imaging.tclean.html): `niter=0` produces the initial dirty/residual image; restored images convolve the deconvolution model with the clean beam and add residuals.

## Bottom line

Do not spend effort cosmetically repairing the current figures first. Preserve any checkpoints for forensic comparison, fix frequency-aware UV coordinates and coordinate/PA conventions, align shared priors, and rerun. Then repair posterior representative selection and produce visibility-domain plus dirty-residual diagnostics before interpreting compact-source sizes or claiming model-selection evidence.

## 1. Repository reconnaissance

### Active workflow

| Role | File | Relevant locations |
|---|---|---|
| Remote launcher | `dynesty_version/launch_headless_scratch.py` | lines 31–34 define input/output/data paths; 60–83 create the job |
| Scratch orchestration | `dynesty_version/launch_fittings_scratch_{bshlog,psrec,pfk}.sh` | core staging and execution at `launch_fittings_scratch_pfk.sh:68–92` |
| Dynesty driver | `dynesty_version/input_dir/agy_run_dynesty.py` | UV ingest 60–98; likelihood 102–130; sampler 181–221; summaries 253–277 |
| Prior transforms | `dynesty_version/input_dir/prior_tform.py` | ring 24–73; 1 blob 80–134; concentric 2-peak 141–211; double-pendulum 218–292; 3 blob 299–360 |
| Image models/Galario calls | `dynesty_version/input_dir/model_prof.py` | primitives 47–148; grid/Galario paths 159–998; dispatch 1004–1067 |
| Posterior/residual diagnostics | `dynesty_version/input_dir/visualize_dynesty.py` | beam conversion 54–86; posterior summary 162–184; UV/residual calculation 213–250; MS write 252–279; `tclean` 283–337; image comparison 341–519 |
| Coordinate/WCS helper | `dynesty_version/input_dir/radec_calc.py` | parameter-to-sky conversion 24–131; model WCS 133–151 |

Supported active models are the 7-parameter ring, 11-parameter ring+blob, 13-parameter concentric two-Gaussian blob, 15-parameter double-pendulum two-peak model, and 19-parameter ring+three-blob model (`model_prof.py:1047–1060`, `1098–1121`).

### Archived workflow

`emcee_version/input_dir/run_mcmc_fit_scratch_pfk_radex.py` is the historical sampler; `model_profiles_pfk_radex.py` and `model_vis_pfk_radex.py` supply its likelihood/model-visibilities, and `visualize_fit_final_radex.py` supplies diagnostics. It carries forward the same 93 GHz conversion (`run_mcmc_fit_scratch_pfk_radex.py:130–135`), coordinate-wise medians (`visualize_fit_final_radex.py:84–101`), 10,000-iteration residual CLEAN (`195–225`), and wrong beam factor (`28–37`). Its 56 walkers are initialized in a tiny `1e-5` ball about one guess (`run_mcmc_fit_scratch_pfk_radex.py:189–203`), which is especially poor for permutation modes.

`dynesty_version/FIX_package/` is not staged or called by the launch path: the core launcher copies `input_dir`, not `FIX_package` (`launch_fittings_scratch_pfk.sh:68–73`). It is therefore not evidence of an executed fix.

## 2. Verification matrix 

The prompt contains ten top-level bullets but calls them eleven points. The matrix splits “best-fit construction” and “circular variables” into separate claims, yielding eleven checks without inventing an omitted claim.

| # | Claim | Code location / direct evidence | Status | Impact |
|---:|---|---|---|---|
| 1 | `xoff = distance*sin(theta)` causes an East–West parity error under Galario conventions. | `model_prof.py:202–204`, `343–345`, `507–509`, `681–683`, `859–861` construct horizontal coordinates increasing with column; blob centers use `+dist*sin(angle)` at `357–358`, `521–522`, `695–700`, `875–883`. Galario requires RA to increase leftward, i.e. decrease with column. | **Confirmed, with qualification** | The angle handedness is reversed. With the subsequent Galario rotation, sky PA is about `PA-theta`. `radec_calc.py:38–42`, `54–58`, `70–82`, `94–114` compensates for this, so the fit is not automatically unusable if its prior covered the source. Parameters and any uncorrected prior interpretation are wrong. |
| 2 | `pars_bf` is assembled from independent marginal medians, not a maximum-likelihood/MAP sample. | `visualize_dynesty.py:176–184` loops over dimensions and assigns each weighted 50th percentile. It never indexes `res.samples[np.argmax(res.logl)]` or another actual joint sample. | **Confirmed** | The vector may lie off the posterior manifold, especially for correlated peak/width/position parameters. Residuals at `244–250` and every model diagnostic use this potentially low-likelihood synthetic point. Nested samples/evidence themselves are unaffected. |
| 3 | Circular/axial variables are summarized and sampled as ordinary linear variables. | Same quantile loop at `176–184`; no circular statistics. Full-circle `Angle 2` prior is `[0,360]` at `prior_tform.py:283–284`; axial ring PA is `[0,180]` at `58–65`; no Dynesty `periodic=` argument at `agy_run_dynesty.py:203–212`. | **Confirmed** | A posterior straddling 0/360 can have a nonsensical median near 180; bounding also treats equivalent endpoints as distant. `[350,390]` is usefully unwrapped, but `[0,360]` is not. Ring PA has 180-degree rather than 360-degree periodicity. |
| 4 | Gaussian widths are linear while peaks are logarithmic, with zero/sub-pixel widths allowed, creating a funnel/flux degeneracy. | `gaussblob_prof` is `10**peak` with linear `sigma` (`model_prof.py:75`, `107`); priors include width zero throughout `prior_tform.py:58–66`, `115–127`, `196`, `263–285`, `333–352`. | **Confirmed** | For a circular blob, integrated flux is proportional to `10**peak * sigma^2`; unresolved data constrain this combination rather than peak and width separately. Near-zero widths are grid-phase dependent and can underflow or produce NaNs at exactly zero. |
| 5 | Component locations are sampled in polar rather than Cartesian coordinates. | Blob priors are `(Dist, Angle)` (`prior_tform.py:123–126`, `197–200`, `275–284`, `341–352`) and transformed with sine/cosine in `model_prof.py`. The ring centre alone is Cartesian `dRA,dDec`. | **Confirmed** | Uniform radius is not uniform area (`p(x,y) proportional to 1/r`), angle is undefined near zero separation, and polar seams complicate sampling. This matters especially for double-pendulum `dist2` including zero. |
| 6 | Multiple Gaussian components have unconstrained permutations/label switching. | In the three-blob prior, blobs 1 and 3 have identical peak/width/distance priors and overlapping angle ranges: `prior_tform.py:341–352`. Blob 2 is angularly separated. Concentric two-peak widths are differentiated (`192`, `196`); double-pendulum roles are hierarchical (`271–284`). Dynesty has no component initialization; historical emcee starts all walkers at one mode (`run_mcmc...py:189–203`). | **Partially True** | Exact/near label exchange is possible for three-blob B1/B3 in the angle overlap, but components are not all unconstrained permutations. The other parameterizations break exchange symmetry, albeit with strong near-coincident degeneracies. |
| 7 | Image grid/pixel sampling is inadequate for narrow components. | `agy_run_dynesty.py:87–96` calls Galario `get_image_size`; actual baseline extrema give `nxy=4096`, `dxy=0.0391567 arcsec`, FOV `160.386 arcsec`. Priors permit `sigma -> 0`; no resolution-aware lower bound or oversampling check exists. | **Partially True** | The grid satisfies Galario's UV-plane Nyquist/FOV heuristic and its centre is correct, but that does not ensure the *model profile* is sampled. Three pixels across FWHM require `sigma >= 0.0499 arcsec`; five require `>=0.0831 arcsec`. Some current priors explicitly enter this unsafe range. Actual posterior widths cannot be checked because outputs are absent. |
| 8 | A scalar 93 GHz wavelength is used despite multi-channel/multi-SPW data. | `agy_run_dynesty.py:87–90`; `visualize_dynesty.py:223–234`; archived emcee equivalents. The input UV table itself says frequency reading failed (external UV table lines 1–3). `M95_C5+C2_cont93_uvtable.listobs.txt:35–40` shows four SPWs centred at 86.3485, 88.2760, 98.3335, and 100.1210 GHz with 8–9 channels; their channel centres span about 85.6376–100.8783 GHz. | **Confirmed** | Relative to the true channel UV coordinate, `u_code/u_true - 1` runs from +8.60% at the low-frequency edge to -7.81% at the high-frequency edge. A size inferred from either edge alone would shift by the reciprocal factors, about -7.92% to +8.47%. Chromatic distortion cannot be repaired by one posterior rescale. This invalidates quantitative fits. |
| 9 | Visibility weights are used as `w=1/sigma^2` in the Gaussian likelihood. | `agy_run_dynesty.py:87`, `95–96` loads column 5; `model_prof.py:271`, `287` (and all other model wrappers) passes it unchanged to `chi2Image`; log likelihood is `-0.5*chi2` at `agy_run_dynesty.py:102–127`. A scan of all 7,258,040 fitted rows found positive finite weights, range 410.315–1666.734. | **Confirmed for ingestion; calibration unverified** | The code-side weighting algebra matches Galario's complex-visibility chi-square. The repository lacks the UV-table exporter/provenance checks, weight-spectrum validation, weight rescaling, or a calibration/jitter term. Statistical errors may therefore be miscalibrated even though weights are multiplied correctly. |
| 10 | Residual imaging applies `niter=10000`, risking CLEANing noise. | `visualize_dynesty.py:306–324` runs Hogbom `tclean` on residual MSs with `niter=10000`; data imaging also uses 10,000 at `286–304`. | **Partially True** | The configured ceiling is confirmed, but “excessive iterations actually executed” cannot be proven without logs: fixed thresholds may stop earlier. A residual diagnostic should first be a dirty image (`niter=0`) under identical gridding/weighting; any residual deconvolution needs justified thresholds/masks and reported iteration counts. |
| 11 | Beam area uses `pi/(2 ln 2)` instead of `pi/(4 ln 2)`. | Docstring gives the correct expression at `visualize_dynesty.py:54–64`, but executable line 83 uses `np.pi/(2*np.log(2))`. Historical emcee uses the same at `visualize_fit_final_radex.py:35`. | **Confirmed** | Beam area is twice too large, so dividing Jy/beam by it makes data and residual surface brightness exactly a factor of two too low. Model Jy/sr is not reduced by this factor, making the comparison internally inconsistent. |

## 3. Independent checks and additional findings

### P0-level additional defects

1. **Frequency information was discarded at extraction.** The fit table header explicitly reads `wavelength[m] = Frequency reading failed for this MS.` It contains only `u, v, Re, Im, weight`; with millions of channel samples but no frequency/SPW/channel field, correct per-row wavelength conversion is impossible downstream. Re-extract from the MS with either UV coordinates already in wavelengths for each channel or an explicit frequency column. The model may also need a spectral-index term across this ~15% fractional bandwidth.

2. **A hidden fixed 15-degree rotation changes the meaning of PA between models.** `model_prof.py:150–153` defines `PA_RAD=15 deg`. All blob-bearing ring models apply it internally (e.g. `350–354`, `514–518`, `688–692`, `866–870`) and then pass the sampled `posangle` to Galario as another rotation (e.g. `444`, `609`, `801`, `986`). The ring-only model does not apply the fixed angle (`206–212`, `287`). Consequently the reported PA is not a common physical parameter across model classes, and log-evidence comparisons do not compare consistently parameterized models.

3. **Shared-parameter priors differ materially between model classes.** Ring-only uses peak/width/radius/inclination/PA ranges `[-5,15]`, `[0,10]`, `[0,20]`, `[0,90]`, `[0,180]` (`prior_tform.py:58–65`), while two-peak models use `[0,8]`, `[0,3]`, `[4,8]`, `[45,80]`, `[-10,10]` (`182–188`, `263–270`). Other blob models use still other shared priors. Evidence is intrinsically prior-dependent; Bayes factors advertised in the README are not interpretable as component-count comparisons unless shared physical parameters and angle conventions have the same justified priors.

4. **The active visualization is guaranteed to raise `NameError`.** `visualize_dynesty.py:561–562` comments out definitions of `cy_clamp` and `cx_clamp`, but lines `590–593` use both. `dynest_radec` returns at least one coordinate even for ring-only (`radec_calc.py:118–122`), so every normal model enters this code. PDF closure at line 636 and the final completion log are never reached. An older file in `FIX_package` defines the variables, but it is not executed.

### Coordinate, origin, and WCS findings

5. **Galario centre is correct; RA handedness is not.** Algebraically, the active grid expression is `(arange(nxy)-nxy/2)*dxy`, placing zero at array index `nxy/2`, exactly Galario's convention for even images. There is no `N/2` versus `N/2+1` bug in the visibility model. The problem is that Galario's RA coordinate must descend with column; using Galario's `get_coords_meshgrid(..., origin='lower')` would avoid the manual sign error. The Dec array increases with row and is correct for `origin='lower'`.

6. **Phase centre matches the MS.** Current hard-coded phase centre `10:43:57.7330 +11:42:12.9996` (`visualize_dynesty.py:108`, `366`; `radec_calc.py:28`) agrees with both fields in `listobs` to substantially below a milliarcsecond in Dec and about 0.0006 arcsec in RA (`listobs:31–34`). There is no evidence here for a phase-centre mismatch. The repository does not document the continuum-subtraction/concatenation phase handling, so that upstream step remains unaudited.

7. **Ring-only coordinate annotation ignores fitted centre offsets.** `radec_calc.py:118–122` returns the phase centre for `twod_gaussring`, even when fitted `dRA,dDec` are nonzero. This affects annotation/cutout placement, not the likelihood.

8. **Plot WCS and model size are brittle.** `visualize_dynesty.py:354–368` sets model size to `data_plot.shape[0] + 1` and invents a separate WCS. For the present pixel scale this appears intended to turn a 919-pixel cutout into an even 920-pixel model, but the invariant is undocumented and not asserted. Use a shared explicit grid/WCS and verify round trips for centre and injected East/North point sources.

### Sampler and posterior findings

9. **The “dynamic” run disables dynamic batches.** `agy_run_dynesty.py:203–221` creates `DynamicNestedSampler(nlive=250, bound='multi', sample='rwalk')` but calls `run_nested(..., maxbatch=0)`. This permits only the baseline batch, defeating the advertised adaptive allocation. Dynesty documents `maxbatch` as the maximum number of added batches.

10. **`nlive=250` and `dlogz_init=0.5` are weak defaults for the 15–19D multimodal cases.** They are not automatically invalid, and `multi`/`rwalk` are defensible choices, but mode die-off and inaccurate evidence must be tested using multiple independent seeds, larger live-point counts, tighter stopping, and a `slice`/`rslice` comparison. No random seed is recorded, so runs are not reproducible. There is no effective posterior sample target.

11. **Checkpointing is not resumption.** A checkpoint filename is supplied (`agy_run_dynesty.py:197–220`), but every invocation constructs a new sampler; it never restores or passes `resume=True`. Interrupted expensive runs are therefore not reliably resumed.

12. **Absolute log evidence lacks the Gaussian normalization.** The likelihood keeps only `-chi2/2`. This is acceptable for Bayes factors among models using exactly the same data and fixed weights because the omitted term is constant, but the printed absolute `ln Z` is not a normalized data probability, and comparisons across changed selections/weights are invalid.

13. **Likelihood allocations are unnecessarily extreme.** Every call rebuilds multiple 4096x4096 coordinate/intermediate arrays; 16 forked workers can demand many tens of GB. Precompute immutable grids and fixed transforms per worker, or use analytic Gaussian Fourier transforms. This is mainly stability/performance but can cause job termination without a scientific diagnostic.

### Residual and unit findings

14. **Intrinsic model and restored CLEAN image are compared directly.** `visualize_dynesty.py:363–375` creates an intrinsic Jy/sr sky model; it is displayed beside the restored data at `379–392` without convolution by the restoring beam. `imsmooth` is imported but unused. Peaks and compact widths therefore cannot be compared visually or via image-plane profiles. Convolve the model with the exact data restoring beam (and apply the same pixel/WCS response), or form all comparisons as forward-modelled dirty/restored products.

15. **Residual cleaning and PB correction hide the most useful diagnostic.** The code should retain the un-deconvolved, un-PB-corrected residual dirty image and PSF for noise statistics. PB correction amplifies noise away from the centre. CASA documents `niter=0` as the dirty/initial residual mode. If a CLEANed residual is also made, keep it secondary and record the stopping reason.

16. **Hard-coded CLEAN mask and thresholds are dataset-specific.** The mask centre `[2048,2052]` and 400-pixel radius (`visualize_dynesty.py:303`, `323`) assume the inferred grid and cover only about 15.66 arcsec radius, while the model priors permit a ring radius of 20 arcsec plus a 10 arcsec width. Thresholds are fixed literals rather than measured multiples of dirty-image RMS.

17. **Residual-MS implantation relies on undocumented flattening order.** `visualize_dynesty.py:257–277` reshapes a flat UV-table vector into the CASA `DATA` column with no check of row/channel mapping. The bundled XX and YY UV tables do have identical UV coordinates in all 7,258,040 rows, and the same model can be subtracted from both polarizations. That does not prove that flat-table order matches CASA column order. Re-extraction should preserve row/channel identifiers and include an exact round-trip assertion before writing.

18. **No weight/calibration diagnostics are produced.** All fitted weights are finite and positive. The first inspected rows show that the combined visibility is the inverse-variance weighted XX/YY mean and its weight is the sum of polarization weights, which is correct. Still required are reduced chi-square by SPW/time/baseline, standardized residual distributions, amplitude-vs-UV-distance and phase residuals, and an optional fractional amplitude calibration nuisance parameter.

19. **Profile nomenclature is ambiguous.** “Width” is actually Gaussian `sigma`, not FWHM (`model_prof.py:52`, `82`). Every table/plot should state sigma explicitly or convert to FWHM using `2.35482*sigma`.

## 4. Assessment of existing results

### What could be salvaged from a checkpoint

- **Coordinate-wise median bug:** fully salvageable by reloading the checkpoint and selecting an actual joint sample. It never changed the likelihood exploration.
- **Beam-area factor, residual CLEAN choice, missing beam convolution, and plotting `NameError`:** salvageable by rerunning post-processing only.
- **Angle handedness:** often salvageable by relabelling angles and regenerating coordinates, conditional on the intended source lying within the *actual mirrored prior support*. Inspect posterior/prior support before accepting this.
- **Hidden 15-degree PA:** potentially salvageable as a deterministic PA reinterpretation for a fixed model class, but not for evidence comparisons across ring-only versus blob-bearing classes with inconsistent priors.

### What requires a new fit

- The per-channel frequency/UV conversion error changes the likelihood differently for every channel and SPW. All quantitative sizes, offsets, compact-component widths/fluxes, residuals, chi-square values, and evidences require refitting from a frequency-aware visibility table.
- Any run whose posterior admits `sigma < 0.05 arcsec` on the present grid must be repeated with an analytic or sufficiently oversampled model and a resolution-aware prior.
- Model-comparison evidences must be recomputed with consistent shared-parameter priors, coordinate conventions, and validated nested-sampling settings.

Because no actual output artifact was available, it is impossible to determine whether a particular run converged, hit a prior boundary, suffered label exchange, or sampled sub-pixel widths. Existing numbers should be treated as exploratory only.

## 5. Ranked action plan

### P0 — blockers / critical correctness

- [ ] **Rebuild the visibility export with frequency retained per datum.** Prefer output columns `(u_lambda, v_lambda, Re, Im, weight, frequency, spw, channel, row-id)` after flags are applied. Never infer all wavelengths from the filename. Assert a few exported rows directly against MS `UVW`, `CHAN_FREQ`, `DATA_DESC_ID`, channel index, `DATA`, `FLAG`, and `WEIGHT_SPECTRUM`/`WEIGHT`.
- [ ] **Refit every scientific model using per-channel `u,v` in wavelengths.** If one achromatic continuum model is retained, justify it; otherwise add a reference-frequency intensity plus spectral index. Compute the grid conservatively from the largest true UV radius.
- [ ] **Replace hand-built coordinates with Galario's coordinate helper or exactly reproduce its descending-RA, ascending-Dec lower-origin grid.** Define one public convention: `dRA>0` East, `dDec>0` North, PA East of North. Then use `x_blob = r*sin(PA)` on the correctly handed RA coordinate. Remove compensating sign algebra from `radec_calc`.
- [ ] **Add deterministic visibility-domain parity tests before refitting.** Inject a single point/Gaussian at +East, -East, +North, -North; compare synthetic visibility phases with the analytic shift theorem and recover the known offsets. Add a nonzero-PA asymmetric source test and a WCS round-trip test.
- [ ] **Remove the hidden `PA_RAD=15 deg` transform.** If 15 degrees is prior knowledge, express it as the same physical PA prior in every model, not a second internal rotation. Ensure ring-only is nested consistently inside every blob model.
- [ ] **Make shared priors identical across models used for evidence comparison.** Document prior scientific rationale and units; rerun evidences. Do not compare old logZ values across the current inconsistent prior volumes.
- [ ] **Repair guaranteed post-processing failure** by defining/clamping `cx_clamp,cy_clamp` and adding a smoke test that completes the PDF/FITS pipeline for every fittype.

### P1 — sampling and numerical stability

- [ ] **Reparameterize each Gaussian as log integrated flux plus log size** (or log FWHM), with a strictly positive, resolution-aware lower size bound. For blobs, `F = 2*pi*I0*sigma^2`; derive the corresponding ring flux parameterization and include inclination consistently.
- [ ] **Use Cartesian East/North offsets for freely located components**, or if polar priors are scientifically desired, use the correct Jacobian (`r^2` uniform for uniform area) and an explicit circular parameter. Never include `r=0` while retaining a meaningful angle.
- [ ] **Break label symmetry deliberately.** Assign non-overlapping spatial regions tied to named sources, or impose a deterministic ordering (with careful treatment at the angular seam). Verify that relabelling leaves the physical posterior unchanged.
- [ ] **Represent circular/axial variables correctly.** Mark truly periodic Dynesty dimensions where supported, keep posterior intervals unwrapped around a chosen mode, and report circular means/credible arcs. Treat disk PA as 180-degree axial data, not a 360-degree direction.
- [ ] **Make the dynamic sampler genuinely dynamic:** remove `maxbatch=0`; set and justify `nlive_init`, `nlive_batch`, a posterior effective-sample target, and tighter evidence tolerance. For 15–19D models, benchmark at least `multi+rwalk` against `multi+slice/rslice`.
- [ ] **Run at least 3 independent recorded seeds** at increasing live-point counts. Require stable posterior summaries and logZ within estimated uncertainties; inspect mode occupancy, effective sample size, call efficiency, bound diagnostics, and prior-edge pile-up.
- [ ] **Implement tested checkpoint resume** and record package versions, seed, input-table checksum, model/prior version, command line, and complete configuration in each output directory.
- [ ] **Select coherent posterior representatives.** For “best fit,” use `res.samples[argmax(res.logl)]` (MAP equals ML only for constant priors in physical coordinates); separately retain the highest quadrature-weight sample if desired and draw multiple posterior predictive samples. Never build a synthetic point from marginal medians.
- [ ] **Precompute model grids/fixed transforms** and reduce per-likelihood memory. Validate any optimization against analytic Gaussian visibilities and the original calculation at well-resolved widths.

### P2 — diagnostics and verification

- [ ] **Generate visibility diagnostics first:** real/imaginary residual vs `u`, `v`, UV distance, frequency/SPW, time, baseline, and polarization; binned complex residuals; standardized residual histogram/Q-Q plot; total and per-subset chi-square.
- [ ] **Make the primary residual image dirty (`tclean niter=0`)** with the same weighting, cell, imsize, phase centre, and data selection as the observed image. Preserve PSF, sum-of-weights, un-PB-corrected residual, and RMS region. Make any CLEANed residual an explicitly secondary product.
- [ ] **Correct beam solid angle** to `Omega_beam = pi/(4 ln 2) * BMAJ * BMIN` in radians squared. Add a unit test using a known Gaussian beam and label FITS `BUNIT` consistently.
- [ ] **Convolve the intrinsic sky model with the exact restoring beam** before image-plane comparisons and 1D profiles. Also provide a model dirty image made through the observed sampling/weighting for apples-to-apples residual checks.
- [ ] **Derive masks and thresholds from WCS/source extent and measured RMS**, not hard-coded pixel locations or absolute values. Record `iterdone`, stopping reason, peak residual, beam, and RMS in the report.
- [ ] **Keep PB-corrected and uncorrected products distinct.** Use uncorrected maps for stationary-noise residual tests; use PB-corrected maps only for surface-brightness presentation within a declared PB cutoff.
- [ ] **Verify MS implantation ordering** with row/channel identifiers and a write-read round trip on a disposable MS outside the repository. Assert XX/YY UV coordinate equality before sharing model visibilities.
- [ ] **Add end-to-end synthetic recovery tests** spanning multiple SPWs, known East/North offsets, a 0/360 angle seam, near-resolution widths, and two exchangeable blobs. Recovery must be correct in both visibilities and WCS plots.

## 6. Quantitative audit notes

- Fitted table rows: 7,258,040 complex samples.
- Projected baseline range: 12.4371–2616.9985 m.
- At the code's assumed 93 GHz: UV radius 3.858–811.831 klambda.
- Galario default grid inferred from those UV coordinates: `4096 x 4096`, `0.0391567 arcsec/pixel`, `160.386 arcsec` FOV; approximate MRS 32.077 arcsec under Galario's heuristic.
- Minimum Gaussian sigma for 3 pixels/FWHM: 0.0499 arcsec; for 5 pixels/FWHM: 0.0831 arcsec.
- Combined-table weights: 410.315–1666.734, all finite/positive. XX weights: 193.230–910.194; YY: 154.867–892.457. All 7,258,040 XX/YY UV coordinates agree row by row.
- SPW centre frequencies: 86.3485, 88.2760, 98.3335, 100.1210 GHz; channel-centre extrema are approximately 85.6376 and 100.8783 GHz. At the SPW centres, `u_true/u_code = frequency/93 GHz` gives 0.9285, 0.9492, 1.0573, and 1.0766 respectively.

## 7. Reference conventions used in this audit

- [Galario image/coordinate specifications](https://mtazzari.github.io/galario/tech-specs.html): RA increases leftward, Dec upward for `origin='lower'`, and the physical origin is pixel `[Nxy/2,Nxy/2]`.
- [Galario Python API](https://mtazzari.github.io/galario/py-api.html): image units are Jy/pixel; `dxy` is radians; UV coordinates are wavelengths; positive `dRA` and `dDec` translate East and North; Galario's chi-square weights both real and imaginary residual terms.
- [Galario coordinate-grid cookbook](https://mtazzari.github.io/galario/cookbook.html): authoritative descending-RA grid construction.
- [Dynesty dynamic sampling documentation](https://dynesty.readthedocs.io/en/v2.1.5/dynamic.html): baseline plus added batches and the meaning of `maxbatch`.
- [CASA `tclean` documentation](https://casadocs.readthedocs.io/en/latest/api/tt/casatasks.imaging.tclean.html): `niter=0` produces the initial dirty/residual image; restored images convolve the deconvolution model with the clean beam and add residuals.

## Bottom line

Do not spend effort cosmetically repairing the current figures first. Preserve any checkpoints for forensic comparison, fix frequency-aware UV coordinates and coordinate/PA conventions, align shared priors, and rerun. Then repair posterior representative selection and produce visibility-domain plus dirty-residual diagnostics before interpreting compact-source sizes or claiming model-selection evidence.
