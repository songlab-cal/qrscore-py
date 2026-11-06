"""QRscore tests (R: QRscore.R)."""

from __future__ import annotations

import logging
import math
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

from ._models import BETA_CAP, ModelFitError, fit_nb_model, fit_zinb
from ._utils import (
    _as_rng, _compute_resample_pval, _compute_s_scores, _gaussian_asymptotic_moments,
    _generate_aa_matrix, _generate_b_matrix, _get_h_inv, _get_h_new, _simplex_sample,
    _weights_at, computeweight_disp, computeweight_mean, get_composition_pvalue, rank_x,
)

logger = logging.getLogger("qrscore")

_ALTERNATIVES = ("two.sided", "greater", "less")


def _require(cond, msg="assertion failed"):
    if not cond:
        raise ValueError(msg)


# --------------------------------------------------------------------------- #
# QRscore_Flex
# --------------------------------------------------------------------------- #
def _flex_one_sample(x, k, p, w, alternative, type, resamp_number, rng):
    _require(not np.any(x < 0), "x must be non-negative")
    t = float(np.sum((x / x.sum()) ** p * w))
    logger.info("The test statistic for the data is %s", t)
    logger.info("Resampling on simplex with %s", resamp_number)
    resampled_ts = _simplex_sample(k, resamp_number, rng) ** p @ w
    return _compute_resample_pval(t, resampled_ts, alternative, type)


def _flex_two_sample(x, y, k, p, w, alternative, approx, type, resamp_number, rng):
    edges = np.concatenate([[-np.inf], np.sort(x), [np.inf]])
    n = y.size
    snk = np.array([np.sum((y >= edges[i]) & (y < edges[i + 1])) for i in range(k)], dtype=float)
    large = n >= 100 and k >= 50
    if approx == "resample" and not large:
        logger.info("Adjusting n and k for resampling")
        t = float(np.sum(((snk + 1) / (n + k)) ** p * w))
    else:
        t = float(np.sum((snk / n) ** p * w))
    logger.info("The test statistic for the data is %s", t)
    _require(k / n >= 1e-3, "k / n must be at least 1e-3")

    if approx == "asymptotic" or large:
        if not large:
            warnings.warn("Sample sizes may be too small...", stacklevel=3)
        logger.info("Applying Gaussian asymptotics...")
        m1, m2 = _gaussian_asymptotic_moments(n, k, w, p)
        z = (t - m1) / math.sqrt(m2)
        if alternative == "two.sided":
            return 2 * min(stats.norm.cdf(z), stats.norm.cdf(-z))
        if alternative == "greater":
            return float(stats.norm.cdf(-z))
        return float(stats.norm.cdf(z))
    logger.info("Using resampling approach with %s", resamp_number)
    return get_composition_pvalue(t, n + k, k, p, w, alternative, type, resamp_number, rng)


def qrscore_flex(x, y=None, p=1, w_list=None, alternative="two.sided", approx="resample",
                 type="unbiased", n_mom=None, resamp_number=5000, rng=None):
    """Flexible non-parametric one- and two-sample tests (R: ``QRscore_Flex``).

    One sample: tests uniformity of the normalised spacings ``x / sum(x)``.
    Two samples: tests whether ``x`` and ``y`` come from the same distribution
    using the weighted power sum of the spacing statistic ``S_{n,k}``.

    Parameters
    ----------
    x, y : array-like
        First and (optional) second sample.
    p : {1, 2}
        Exponent of the test statistic.
    w_list : array-like
        Weights; length ``len(x)`` (one sample) or ``len(x) + 1`` (two samples).
    alternative : {"two.sided", "greater", "less"}
    approx : {"resample", "asymptotic"}
        Gaussian asymptotics are used automatically when n >= 100 and k >= 50.
    type : {"unbiased", "valid", "both"}
        Resampling p-value flavour (Hemerik & Goeman, 2018).
    n_mom : unused, kept for API parity with R.
    resamp_number : int
        Number of compositions / simplex vectors to draw.
    rng : numpy.random.Generator, int or None

    Returns
    -------
    float, or dict with "unbiased" and "valid" when ``type="both"`` and
    resampling is used.
    """
    rng = _as_rng(rng)
    x = np.asarray(x, dtype=float)
    w = np.asarray(w_list, dtype=float)
    k = x.size + 1 if y is not None else x.size
    if y is not None:
        _require(w.size == k, "w_list must have length len(x) + 1")
    _require(p in (1, 2), "p must be 1 or 2")
    _require(approx in ("resample", "asymptotic"), "approx must be 'resample' or 'asymptotic'")
    w = w / w.max()
    logger.info("Normalizing weight vector...")
    if y is not None:
        return _flex_two_sample(x, np.asarray(y, dtype=float), k, p, w, alternative,
                                approx, type, resamp_number, rng)
    return _flex_one_sample(x, k, p, w, alternative, type, resamp_number, rng)


# --------------------------------------------------------------------------- #
# Model fitting used for weights
# --------------------------------------------------------------------------- #
def _estimate_nb_parameters(counts, offset, fit_method):
    logger.info("Estimating NB parameters.")
    fit = fit_nb_model(counts, offset, fit_method)
    beta = np.minimum(fit.theta, BETA_CAP)       # NaN propagates, as min() in R
    return {"beta": float(beta), "mu": fit.mu, "pi": 0.0}


def _estimate_zinb_parameters(counts, offset, fit_method):
    logger.info("Estimating ZINB parameters.")
    if offset is None:
        fit = fit_zinb(counts, None, fit_method)
    else:
        try:
            fit = fit_zinb(counts, offset, fit_method)
        except (ModelFitError, FloatingPointError, ValueError):
            logger.info("Offset model failed, using normalized counts.")
            fit = fit_zinb(np.round(counts / np.exp(offset)), None, fit_method)
    return {"fit": fit, "beta": float(min(fit.theta, BETA_CAP)), "mu": fit.mu, "pi": fit.pi}


def _fit_and_validate_zinb(combined, offset, gene_name, pi_threshold, lr_test, fit_method):
    zinb = _estimate_zinb_parameters(combined, offset, fit_method)
    if lr_test:
        logger.info("Applying likelihood ratio test.")
        nb = fit_nb_model(combined, offset, fit_method)
        if nb.normalized:
            raise ValueError("NB offset model failed; likelihood ratio test not available.")
        lrt_p = stats.chi2.sf(max(0.0, 2 * (zinb["fit"].loglik - nb.loglik)), df=1)
        if lrt_p >= 0.05:
            return _estimate_nb_parameters(combined, offset, fit_method)
    if zinb["pi"] > pi_threshold:
        msg = (f"The estimated probability of zeros is > {pi_threshold:.2f}." if gene_name is None
               else f"The estimated probability of zeros for gene {gene_name} is > {pi_threshold:.2f}.")
        warnings.warn(f"{msg} Returning NA.", stacklevel=3)
        return None
    return zinb


# --------------------------------------------------------------------------- #
# QRscore_ZINB
# --------------------------------------------------------------------------- #
def qrscore_zinb(x, y, size_factors=None, zero_inflation=True, lr_test=False,
                 approx="resample", alternative="two.sided", resamp_num=20000,
                 pi_threshold=0.95, gene_name=None, measure="mean", p_value=True,
                 fit_method="r", rng=None):
    """Two-sample test for mean or dispersion shifts in (ZI)NB data (R: ``QRscore_ZINB``).

    Weights are the score function of a NB or ZINB model fitted to the pooled
    sample. Returns a p-value (``nan`` when the ZINB zero probability exceeds
    ``pi_threshold``), or, with ``p_value=False``, a dict with ``weight`` and ``var``.

    ``fit_method="r"`` (default) fits the NB / ZINB models exactly as
    ``MASS::glm.nb`` / ``pscl::zeroinfl`` do, so results match the R package;
    ``"mle"`` uses true maximum likelihood estimates instead (see README).
    """
    rng = _as_rng(rng)
    x = np.asarray(x, dtype=float)
    _require(y is not None, "y should not be NULL.")
    y = np.asarray(y, dtype=float)
    if np.any(np.round(x) != x) or np.any(np.round(y) != y):
        warnings.warn("Input values are not integers. Round input.", stacklevel=2)
        x, y = np.round(x), np.round(y)
    _require(np.all(x >= 0) and np.all(y >= 0), "Input contains negative values.")

    n, k = y.size, x.size
    alpha = k / (n + k)
    combined = np.concatenate([x, y])

    if size_factors is not None:
        size_factors = np.asarray(size_factors, dtype=float)
        if size_factors.size != combined.size:
            raise ValueError("Length of size_factors must match length(x) + length(y).")
        offset = np.log(size_factors)
        x_norm, y_norm = x / size_factors[:k], y / size_factors[k:]
    else:
        offset, x_norm, y_norm = None, x, y

    if np.sum(combined == 0) == 0 or not zero_inflation:
        est = _estimate_nb_parameters(combined, offset, fit_method)
    else:
        est = _fit_and_validate_zinb(combined, offset, gene_name, pi_threshold, lr_test, fit_method)
        if est is None:
            return np.nan

    if measure == "dispersion":
        results = computeweight_disp(est["beta"], est["mu"], est["pi"], n, k, rng=rng)
    else:
        results = computeweight_mean(est["beta"], est["mu"], est["pi"], n, k)
    if not p_value:
        return results

    weights = results["weight"]
    est_sd = math.sqrt(alpha * (1 - alpha) * results["var"])
    zscore = float(np.mean(_weights_at(weights, rank_x(x_norm, y_norm, rng=rng))))

    if approx == "asymptotic" or (n >= 100 and k >= 50):
        zscore = k / math.sqrt(n + k) * zscore
        if alternative == "two.sided":
            return float(2 * stats.norm.sf(abs(zscore), scale=est_sd))
        if alternative == "greater":
            return float(stats.norm.sf(zscore, scale=est_sd))
        return float(stats.norm.cdf(zscore, scale=est_sd))

    if n < 10 and k < 10:
        logger.info("Using exact enumeration")
        idx = np.array(list(_combinations(n + k, k)))
        null_ts = weights[idx].mean(axis=1)
        p_exact = np.mean(null_ts < zscore)
        pv = _tail(p_exact, alternative)
        return float(1 / idx.shape[0]) if pv == 0 else float(pv)

    logger.info("Using resampling with %s", resamp_num)
    null_ts = np.empty(resamp_num)
    chunk = max(1, 4_000_000 // (n + k))
    for start in range(0, resamp_num, chunk):
        m = min(chunk, resamp_num - start)
        idx = np.argpartition(rng.random((m, n + k)), k - 1, axis=1)[:, :k]
        null_ts[start:start + m] = weights[idx].mean(axis=1)
    return float(_tail(np.mean(null_ts < zscore), alternative))


def _combinations(n, k):
    import itertools
    return itertools.combinations(range(n), k)


def _tail(cdf, alternative):
    if alternative == "two.sided":
        return 2 * min(cdf, 1 - cdf)
    if alternative == "greater":
        return 1 - cdf
    return cdf


# --------------------------------------------------------------------------- #
# QRscore_ZINB_nSamples
# --------------------------------------------------------------------------- #
def qrscore_zinb_nsamples(samples, labels, size_factors=None, zero_inflation=True,
                          lr_test=False, approx="resample", resamp_num=20000,
                          pi_threshold=0.95, gene_name=None, measure="mean",
                          perturb=True, fit_method="r", rng=None):
    """K-sample chi-square test for mean or dispersion differences (R: ``QRscore_ZINB_nSamples``).

    Groups are taken in order of first appearance in ``labels``. ``approx``,
    ``resamp_num`` and ``perturb`` are accepted for API parity; the K-sample
    test is always asymptotic (chi-square with K - 1 df).
    """
    rng = _as_rng(rng)
    samples = np.asarray(samples, dtype=float)
    labels = np.asarray(labels)
    unique_labels = pd.unique(labels)
    n_groups = unique_labels.size
    sample_list = [samples[labels == lab] for lab in unique_labels]
    if size_factors is not None:
        size_factors = np.asarray(size_factors, dtype=float)
        norm = samples / size_factors
        norm_sample_list = [norm[labels == lab] for lab in unique_labels]
        size_factors = np.concatenate([size_factors[labels == lab] for lab in unique_labels])
    else:
        norm_sample_list = sample_list

    group_sizes = np.array([s.size for s in sample_list])
    n_all = group_sizes.sum()
    b_matrix = _generate_b_matrix(n_groups)
    props = group_sizes / n_all
    aa_matrix = _generate_aa_matrix(props)

    x = sample_list[0]
    y = np.concatenate(sample_list[1:])
    results = qrscore_zinb(x, y, size_factors=size_factors, zero_inflation=zero_inflation,
                           lr_test=lr_test, approx=approx, resamp_num=resamp_num,
                           pi_threshold=pi_threshold, gene_name=gene_name, measure=measure,
                           p_value=False, fit_method=fit_method, rng=rng)
    if not isinstance(results, dict):
        return np.nan

    s_scores = _compute_s_scores(norm_sample_list, results["weight"], n_all, rng)
    test_vec = b_matrix.T @ np.diag(1 / np.sqrt(props)) @ s_scores
    h_inv = _get_h_inv(_get_h_new(results["var"], b_matrix, aa_matrix, gene_name), gene_name)
    if h_inv is None:
        return np.nan
    q_all = float(test_vec @ h_inv @ test_vec)
    return float(stats.chi2.sf(q_all, df=n_groups - 1))


# --------------------------------------------------------------------------- #
# Mann–Whitney, matching R's wilcox.test(correct = FALSE)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _wilcox_counts(m, n):
    """Counts of the Mann–Whitney U statistic for sample sizes m, n."""
    # f(m, n)[u] = f(m-1, n)[u-n] + f(m, n-1)[u]
    if m == 0 or n == 0:
        return np.ones(1)
    a, b = _wilcox_counts(m - 1, n), _wilcox_counts(m, n - 1)
    out = np.zeros(m * n + 1)
    out[n:n + a.size] += a
    out[:b.size] += b
    return out


def _pwilcox(q, m, n, lower_tail=True):
    c = _wilcox_counts(m, n)
    q = int(math.floor(q + 1e-7))
    lo = c[: max(0, min(q, m * n)) + 1].sum() / c.sum() if q >= 0 else 0.0
    return lo if lower_tail else 1 - lo


def wilcox_test_pvalue(x, y, alternative="two.sided"):
    """p-value of R ``wilcox.test(x, y, correct = FALSE)`` (two-sample)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx, ny = x.size, y.size
    r = stats.rankdata(np.concatenate([x, y]))
    statistic = r[:nx].sum() - nx * (nx + 1) / 2
    ties = np.unique(r).size != r.size
    if nx < 50 and ny < 50 and not ties:
        if alternative == "two.sided":
            pv = (_pwilcox(statistic - 1, nx, ny, lower_tail=False) if statistic > nx * ny / 2
                  else _pwilcox(statistic, nx, ny))
            return min(2 * pv, 1.0)
        if alternative == "greater":
            return _pwilcox(statistic - 1, nx, ny, lower_tail=False)
        return _pwilcox(statistic, nx, ny)
    _, nties = np.unique(r, return_counts=True)
    z = statistic - nx * ny / 2
    sigma = math.sqrt(nx * ny / 12 * ((nx + ny + 1)
                                      - np.sum(nties ** 3 - nties) / ((nx + ny) * (nx + ny - 1))))
    z /= sigma
    if alternative == "two.sided":
        return float(2 * min(stats.norm.cdf(z), stats.norm.sf(z)))
    if alternative == "greater":
        return float(stats.norm.sf(z))
    return float(stats.norm.cdf(z))


# --------------------------------------------------------------------------- #
# QRscoreTest
# --------------------------------------------------------------------------- #
def _run_one_sample(samples, p, w_list, alternative, approx, type, n_mom, resamp_number, rng):
    logger.info("Performing a one-sample test.")
    _require(p is not None, "Exponent p must be provided.")
    _require(w_list is not None, "wList must be provided.")
    _require(len(samples) == len(w_list), "wList and sample lengths must match.")
    _require(p > 0 and abs(p - round(p)) < np.finfo(float).eps ** 0.5, "Only integer p supported.")
    _require(approx == "resample", "Only 'resample' supported.")
    return qrscore_flex(samples, None, p, w_list, alternative, approx, type, n_mom, resamp_number, rng)


def _run_two_sample(sample_list, unique_labels, alternative, approx, resamp_number,
                    zero_inflation, lr_test, pi_threshold, gene_name, measure, p, w_list,
                    type, n_mom, use_base_r, size_factor_list, fit_method, rng):
    x, y = sample_list
    x_label, y_label = unique_labels
    sf_x, sf_y = size_factor_list if size_factor_list is not None else (None, None)
    if x.size > y.size:
        # The smaller group is always x. Size factors are swapped together with
        # the samples (the R code does not swap them; see README).
        x, y, x_label, y_label, sf_x, sf_y = y, x, y_label, x_label, sf_y, sf_x
    size_factors = None if size_factor_list is None else np.concatenate([sf_x, sf_y])
    logger.info("Performing a two-sample test. '%s' is x and '%s' is y.", x_label, y_label)

    if w_list is None or p is None:
        logger.info("Use ZINB or NB model to estimate weights.")
        return qrscore_zinb(x, y, size_factors, zero_inflation, lr_test, approx, alternative,
                            resamp_number, pi_threshold, gene_name, measure,
                            fit_method=fit_method, rng=rng)

    w_list = np.asarray(w_list, dtype=float)
    _require(x.size + 1 == w_list.size, "wList must have length = length(x) + 1.")
    _require(p > 0 and abs(p - round(p)) < np.finfo(float).eps ** 0.5, "Only integer p supported.")
    _require(approx == "resample", "Only 'resample' supported.")
    is_mann_whitney = np.array_equal(w_list, np.arange(x.size, -1, -1)) and p == 1
    if is_mann_whitney and use_base_r:
        logger.info("Using Mann-Whitney test (as R wilcox.test)...")
        return wilcox_test_pvalue(x, y, alternative)
    logger.info("Using QRscore_Flex with given weights.")
    if size_factor_list is not None:
        x, y = x / sf_x, y / sf_y
    return qrscore_flex(x, y, p, w_list, alternative, approx, type, n_mom, resamp_number, rng)


def qrscore_test(samples, labels=None, size_factors=None, p=None, w_list=None,
                 alternative="two.sided", approx="resample", type="unbiased", n_mom=100,
                 resamp_number=5000, zero_inflation=True, lr_test=False, pi_threshold=0.95,
                 gene_name=None, measure="mean", perturb=True, use_base_r=True,
                 fit_method="r", rng=None):
    """One-, two- or K-sample QRscore test (R: ``QRscoreTest``).

    * No labels (or a single label): one-sample uniformity test; needs ``p`` and ``w_list``.
    * Two groups: ``qrscore_flex`` with the given ``p``/``w_list``, or, if either
      is ``None``, ``qrscore_zinb`` with model-based weights for ``measure``
      ("mean" or "dispersion").
    * More groups: ``qrscore_zinb_nsamples``.

    Groups are ordered by sorted label (as ``sort(unique(labels))`` in R), and in
    the two-sample case the smaller group becomes ``x``.
    Any error is converted to a warning and ``nan`` is returned, as in R.
    """
    rng = _as_rng(rng)
    try:
        _require(samples is not None, "Samples cannot be NULL.")
        _require(alternative in _ALTERNATIVES, "Invalid alternative value.")
        _require(approx in ("resample", "asymptotic"), "Invalid approximation method.")
        _require(type in ("unbiased", "valid", "both"), "Invalid type value.")
        samples = np.asarray(samples, dtype=float)
        if labels is None or pd.unique(np.asarray(labels)).size == 1:
            return _run_one_sample(samples, p, w_list, alternative, approx, type, n_mom,
                                   resamp_number, rng)
        labels = np.asarray(labels)
        _require(labels.size == samples.size, "Sample and label lengths must match.")
        unique_labels = np.sort(pd.unique(labels))
        sample_list = [samples[labels == lab] for lab in unique_labels]
        size_factor_list = None
        if size_factors is not None:
            size_factors = np.asarray(size_factors, dtype=float)
            size_factor_list = [size_factors[labels == lab] for lab in unique_labels]
        if unique_labels.size == 2:
            return _run_two_sample(sample_list, unique_labels, alternative, approx, resamp_number,
                                   zero_inflation, lr_test, pi_threshold, gene_name, measure, p,
                                   w_list, type, n_mom, use_base_r, size_factor_list,
                                   fit_method, rng)
        logger.info("Performing multi-sample test. Use ZINB or NB model to estimate the weights.")
        return qrscore_zinb_nsamples(samples, labels, size_factors, zero_inflation, lr_test,
                                     approx, resamp_number, pi_threshold, gene_name, measure,
                                     perturb, fit_method=fit_method, rng=rng)
    except Exception as e:  # noqa: BLE001 - mirrors R's tryCatch(error = ...)
        warnings.warn(f"QRscoreTest failed: {e}", stacklevel=2)
        return np.nan
