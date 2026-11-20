# Uncertainty: what a point carries, how its bar and its fit read it

One rule decides every uncertainty in v3: **a reduced point carries the
sufficient statistic of its samples, and its error bar, its published sigma
and its fit likelihood are all read from that one object by the samples'
sampling model.**  Nothing estimates the uncertainty of a parameter (a mean,
a proportion, a rate, a fitted parameter) from the scatter of data without
the model that connects them.

## 1. Reduced points

| Samples reduced by MEAN | Carried | Point | Error bar (68.27 %) | Fit likelihood |
|---|---|---|---|---|
| boolean (Bernoulli trials) | successes `k = y n`, trials `n` | `k / n` | Wilson score interval, z = 1 | binomial, `n` trials |
| real, `n > 1` | mean, centred sum of squares, `n` | mean | `mean ± s/sqrt(n)` (`s` with `n - 1`) | Student-t, `nu = n - 1` |
| real, `n = 1` with a stated sigma | value, sigma | value | `value ± sigma` | Gaussian, sigma known |
| histogram counts | counts per bin | -- | -- | Poisson |

A proportion is never given the Gaussian standard error
`sqrt(p (1 - p) / (n - 1))`: it is a function of the estimate itself, it is
zero at `k = 0` and `k = n`, and used as a fit weight it makes the points
that happened to come out at 0 or 1 the heaviest of the curve.

### Wilson score interval at z = 1

    c = (k + 1/2) / (n + 1)
    h = sqrt(k (n - k) / n + 1/4) / (n + 1)
    interval = [c - h, c + h]

It lies in `[0, 1]`, always contains `k / n`, and has width `1 / (n + 1)`
at `k = 0` or `k = n` (never zero).  `h` is the **standard uncertainty of a
proportion** wherever one symmetric number is needed (a published sigma, a
relative error).  One implementation: `zlc_core::scientific::statistics`
(`proportion_interval`, `proportion_uncertainty`); Python reaches it through
the native service `scientific.proportion` (no Python copy).

### Pooling

Proportions pool as trials: a window or group of shots is `sum k / sum n`
with `sum n` trials -- every loaded site one trial, so a shot that loaded
three atoms counts less than one that loaded thirty.  Real means pool their
moments (Chan); a window's sem is that of the pooled samples.

## 2. Fits

Per point, the likelihood is the one the table above names.

* **Binomial** (observations with `trials`): deviance
  `d = 2 n [ y ln(y/f) + (1 - y) ln((1 - y)/(1 - f)) ]` (a `0 ln 0` term is
  0), exact on `f` in `[0, 1]` and infinite outside it -- a probability past
  1 has no likelihood; residual `sign(f - y) sqrt(d)`; the solve minimises
  `sum d / 2`, the negative log-likelihood up to a constant, and a step that
  would leave `[0, 1]` slides along its edge (an MLE on the edge -- all
  successes at `t = 0` -- is common).
* **Poisson** (histogram targets): unchanged deviance (`f` floored at
  `COUNT_FLOOR`), infinite for `f < 0`.
* **Student-t** for means of few repeats, **Gaussian** for a stated or no sigma: unchanged.

### Covariance: scale only what the model does not fix

* Binomial and Poisson: the noise is fixed by the model (`f (1 - f) / n`,
  `f`), nothing is rescaled by deviance / dof.  Each free parameter's
  uncertainty is its 68.27 % profile-likelihood interval (MINOS): the values
  where the deviance, minimised over the other parameters, rises at most 1;
  a side that does not rise by 1 before a solver bound or the edge of the
  support (a model value leaving `[0, 1]`, or going negative) ends there,
  one-sided -- a proportion fitted at 0 or 1 is such an edge, where the
  log-likelihood is linear and no curvature describes it.  The standard
  error is the interval's half-width, the covariance those half-widths on
  the correlation of the inverse Fisher information
  (`(sum J J^T n / (f (1 - f)))^+`, `(sum J J^T / f)^+`), so its diagonal is
  the errors squared.  The deviance per degree of freedom is still reported
  (as the reduced chi-square) for judging the model.
* Student-t: the M-estimator sandwich times `n / (n - p)`, unchanged.
* Gaussian with a stated sigma or none: `(J^T W J)^+` times the reduced
  chi-square (its floor at the rounding scatter), unchanged.  A stated sigma
  is not known to be absolute (published sigmas carry no degrees of
  freedom, section 3), so the residuals set the scale.

Every fit result says which basis its errors have (`ErrorBasis`), and the fit
annotation prints it beside the parameters: the likelihood's (`binomial`,
`Poisson`), `Student-t, nu = n - p`, or `scaled by chi2_r, nu = n - p`.  An
error with finite `nu` is itself estimated: its 68 % interval is
`t_{nu}`-wide, not normal.

Nested-model evidence (BIC) for a likelihood with no free scale is
`(D_one - D_two) - (k_two - k_one) ln N` with `D` the deviance -- binomial
and Poisson alike.

## 3. Publication

A published dataset carries values, validity and one sigma per sample: a
stated standard uncertainty.  Derive publishes a mean of reals with its sem,
a proportion with `h` (section 1), a fitted parameter with its standard error.
Within one Derive expression the full statistic travels (`Operand.dof`,
`Operand.trials`), so `mean` then `fit` in one expression is fitted by the
right likelihood.  A panel reducing the raw booleans itself does the same.
Downstream of a publication, a sigma is a stated one (section 2).

## 4. Summaries

Proportions reported by nodes (rearrangement occupancy, SLM-feedback loading
rate, calibration fidelities) use the same Wilson interval and `h` through
`scientific.proportion`.  A calibration's per-site fidelity is measured on the
samples that trained it (not held out); its interval is the counting
interval of that measurement.
