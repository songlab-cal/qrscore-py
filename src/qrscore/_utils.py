"""Statistical building blocks of QRscore (R: utils.R)."""

from __future__ import annotations

import itertools
import logging
import math
import warnings

import numpy as np
from scipy import special

from ._distributions import dzinbinom, qzinbinom, rzinbinom

logger = logging.getLogger("qrscore")


def _as_rng(rng):
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


# --------------------------------------------------------------------------- #
# Ranks and spacings
# --------------------------------------------------------------------------- #
def rank_x(x, y=None, ties_break=True, rng=None):
    """Ranks of the sorted ``x`` values within the merged sample ``c(x, y)``.

    Mirrors R ``rank_x``: ties are broken by adding Uniform(0, gap/2) noise,
    where ``gap`` is the smallest positive difference in the merged sample.
    Returns 1-based ranks of ``sort(x)``.
    """
    rng = _as_rng(rng)
    x = np.asarray(x, dtype=float)
    y = np.empty(0) if y is None else np.asarray(y, dtype=float)
    if ties_break:
        diffs = np.diff(np.sort(np.concatenate([x, y])))
        pos = diffs[diffs > 0]
        if pos.size == 0:
            # R: min() of an empty vector is Inf and runif(0, Inf) is NaN,
            # so every rank becomes NA.
            warnings.warn("all values are identical; ranks are undefined", stacklevel=2)
            return np.full(x.size, np.nan)
        gap = pos.min()
        x = x + rng.uniform(0, gap / 2, x.size)
        y = y + rng.uniform(0, gap / 2, y.size)
    x = np.sort(x)
    combined = np.sort(np.concatenate([x, y]))
    return np.searchsorted(combined, x, side="left").astype(float) + 1


def _weights_at(weights, ranks):
    ranks = np.asarray(ranks, dtype=float)
    if np.any(np.isnan(ranks)):
        return np.full(ranks.shape, np.nan)
    return np.asarray(weights)[ranks.astype(int) - 1]


# --------------------------------------------------------------------------- #
# Integer compositions and the simplex
# --------------------------------------------------------------------------- #
def _all_compositions(n, k):
    """All compositions of ``n`` into ``k`` positive parts (arrangements::compositions)."""
    if k == 1:
        return np.array([[n]], dtype=float)
    ncomb = math.comb(n - 1, k - 1)
    cuts = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(1, n), k - 1)),
                       dtype=np.int64, count=ncomb * (k - 1)).reshape(ncomb, k - 1)
    bounds = np.hstack([np.zeros((ncomb, 1), np.int64), cuts, np.full((ncomb, 1), n, np.int64)])
    return np.diff(bounds, axis=1).astype(float)


def _sample_compositions(n, k, nsample, rng):
    """Uniform random compositions of ``n`` into ``k`` positive parts."""
    if k == 1:
        return np.full((nsample, 1), float(n))
    out = np.empty((nsample, k))
    chunk = max(1, 4_000_000 // max(n - 1, 1))
    for start in range(0, nsample, chunk):
        m = min(chunk, nsample - start)
        keys = rng.random((m, n - 1))
        cuts = np.sort(np.argpartition(keys, k - 2, axis=1)[:, : k - 1], axis=1) + 1
        bounds = np.hstack([np.zeros((m, 1), np.int64), cuts, np.full((m, 1), n)])
        out[start:start + m] = np.diff(bounds, axis=1)
    return out


def _simplex_sample(k, nsample, rng):
    """Uniform samples on the (k-1)-simplex (hitandrun::simplex.sample)."""
    return rng.dirichlet(np.ones(k), size=nsample)


# --------------------------------------------------------------------------- #
# p-value helpers
# --------------------------------------------------------------------------- #
def _compute_exact_pval(t, exact_ts, alternative):
    if alternative == "two.sided":
        return 2 * min(np.mean(exact_ts <= t), np.mean(exact_ts >= t))
    if alternative == "greater":
        return float(np.mean(exact_ts >= t))
    return float(np.mean(exact_ts <= t))


def _compute_resample_pval(t, resampled_ts, alternative, type):
    ext = np.append(resampled_ts, t)
    cdf = np.mean(resampled_ts < t)
    cdf_upp = 1 - np.mean(ext >= t)
    cdf_low = np.mean(ext <= t)
    if alternative == "two.sided":
        unbiased, valid = 2 * min(cdf, 1 - cdf), 2 * min(cdf_low, 1 - cdf_upp)
    elif alternative == "greater":
        unbiased, valid = 1 - cdf, 1 - cdf_upp
    else:
        unbiased, valid = cdf, cdf_low
    if type == "unbiased":
        return float(unbiased)
    if type == "valid":
        return float(valid)
    return {"unbiased": float(unbiased), "valid": float(valid)}


def get_composition_pvalue(t, n, k, p, w_list, alternative="two.sided",
                           type="unbiased", resamp_number=5000, rng=None):
    """Approximate p-value by resampling integer compositions (R: getCompositionPValue).

    Exact enumeration of all k-compositions of n is used when n <= 40 and k <= 10.
    """
    w = np.asarray(w_list, dtype=float)
    if n <= 40 and k <= 10:
        logger.info("n and k are small enough, computing exact p-value...")
        exact_ts = (_all_compositions(n, k) / n) ** p @ w
        return _compute_exact_pval(t, exact_ts, alternative)
    comps = _sample_compositions(n, k, resamp_number, _as_rng(rng))
    resampled_ts = (comps / n) ** p @ w
    return _compute_resample_pval(t, resampled_ts, alternative, type)


def _gaussian_asymptotic_moments(n, k, w, p):
    """Null mean and variance of the spacing statistic (R: .get_gaussion_asymptotics_moments)."""
    w = np.asarray(w, dtype=float)
    if p == 1:
        m1 = w.sum() / k
        quad = k * np.sum(w ** 2) - w.sum() ** 2            # w' (kI - 11') w
        m2 = ((k / n + 1) / (k ** 2 * (k + 1))) * quad
    else:
        m1 = ((2 + k / n - 1 / n) / ((k + 1) * k)) * w.sum()
        w2 = np.sum(w ** 2)
        denom = k ** 2 * (1 + k) ** 2 * (2 + k) * (3 + k)
        coef1 = ((k - 1) * (k / n + 1) * (2 + k / n - 1 / n)
                 * (12 - 6 / n + k * (k / n + 10 - 5 / n)) / denom)
        offdiag = w.sum() ** 2 - w2
        coef2 = ((k / n + 1) * (6 / n ** 2 + k * (3 + k * (k - 2)) / n ** 2 - 24 / n
                                + 8 * (k - 1) * k / n + 8 * (3 + 2 * k)) / denom)
        m2 = w2 * coef1 - offdiag * coef2
    return m1, m2


# --------------------------------------------------------------------------- #
# Weights from the (ZI)NB score functions
# --------------------------------------------------------------------------- #
def _log1pmx(t):
    """log(1 + t) - t, accurate for small |t| (series where direct evaluation cancels)."""
    t = np.asarray(t, dtype=float)
    with np.errstate(all="ignore"):
        out = np.log1p(t) - t
    small = np.abs(t) < 0.1
    if np.any(small):
        j = np.arange(2, 41)
        ts = t[small][..., None]
        out = np.array(out, copy=True)
        out[small] = np.sum((-1.0) ** (j + 1) * ts ** j / j, axis=-1)
    return out


def _nb_zero_share(beta, log_r, pi):
    """(1 - pi) r^beta / ((1 - pi) r^beta + pi), evaluated in log space."""
    if pi <= 0:
        return 1.0
    return float(special.expit(np.log1p(-pi) + beta * log_r - np.log(pi)))


def computeweight_mean(beta, mu, pi, n, k):
    """Weights targeting a mean shift in a ZINB model (R: computeweight_mean).

    The weights are the score of the ZINB log-likelihood for a change in log(mu)
    at the quantiles of the fitted ZINB. Returns a dict with ``weight`` (length
    n + k, centred) and ``var``, the exact variance of the score.
    """
    gdict = qzinbinom(np.arange(1, n + k + 1) / (n + k + 1), size=beta, mu=mu, pi=pi)
    u = mu / (beta + mu)
    r = 1 - u
    rb = np.exp(beta * np.log1p(-u))                # r^beta, accurate for large beta
    w0 = _nb_zero_share(beta, np.log1p(-u), pi)     # (1 - pi) r^beta / P(X = 0)
    zvals = np.where(gdict == 0, -mu * r * w0, r * (gdict - mu))
    # The weights are the score of the ZINB log-likelihood for a change in log(mu),
    # so E[z(X)] = 0 and the variance is E[z(X)^2].
    term1 = beta * mu / (beta + mu)
    test_var = (1 - pi) * (term1 - mu ** 2 * r ** 2 * rb) + mu ** 2 * (1 - pi) * r ** 2 * rb * w0
    if not np.isfinite(test_var) or test_var == 0:
        logger.info("Using weight dictionary to calculate variance.")
        test_var = np.var(zvals, ddof=1)
    return {"weight": zvals - zvals.mean(), "var": float(test_var)}


def _z_function_disp(x, beta, mu, pi):
    """Score of the ZINB log-likelihood for a dispersion change (size = theta * beta)
    at theta = 1, vectorised over integer counts (R: .z_function_disp).

    Uses digamma(beta + x) - digamma(beta) = sum_{j<x} 1/(beta + j) so the
    expression has no cancellation and stays accurate for large beta.
    """
    x = np.asarray(x, dtype=float)
    xi = x.astype(np.int64)
    u = mu / (beta + mu)
    l1 = float(_log1pmx(-u))
    log_r = np.log1p(-u)
    xmax = int(xi.max()) if xi.size else 0
    if xmax > 0:
        j = np.arange(xmax, dtype=float)
        S = np.concatenate([[0.0], np.cumsum((mu - j) / ((beta + j) * (beta + mu)))])
    else:
        S = np.zeros(1)
    z = beta * (S[xi] + l1)
    return np.where(xi == 0, beta * l1 * _nb_zero_share(beta, log_r, pi), z)


def computeweight_disp(beta, mu, pi, n, k, tail=1e-4, big_n=10 ** 6, rng=None):
    """Weights targeting a dispersion shift in a ZINB model (R: computeweight_disp).

    Returns a dict with ``weight`` (length n + k, centred) and ``var``.
    For ``mu > 1e4`` the score variance is estimated from ``big_n`` random draws.
    """
    gdict = qzinbinom(np.arange(1, n + k + 1) / (n + k + 1), size=beta, mu=mu, pi=pi)
    if mu > 1e4:
        data = rzinbinom(int(big_n), size=beta, mu=mu, pi=pi, rng=_as_rng(rng))
        x_values, counts = np.unique(data, return_counts=True)
        x_prob = counts / big_n
    else:
        tail_q = int(qzinbinom(1 - tail, size=beta, mu=mu, pi=pi))
        x_values = np.arange(tail_q + 1, dtype=float)
        x_prob = dzinbinom(x_values, size=beta, mu=mu, pi=pi)

    z_x = _z_function_disp(x_values, beta, mu, pi)
    e_z = np.sum(z_x * x_prob)
    h_value = np.sum((z_x - e_z) ** 2 * x_prob)

    zvals = _z_function_disp(gdict, beta, mu, pi)
    weight = zvals - zvals.mean()
    if np.any(np.isnan(weight)):
        raise ValueError("Weight cannot be calculated.")
    if not np.isfinite(h_value) or h_value == 0:
        logger.info("Using weight dictionary to calculate variance.")
        var = float(np.var(zvals, ddof=1))
    else:
        var = float(h_value)
    return {"weight": weight, "var": var}


# --------------------------------------------------------------------------- #
# Multi-sample helpers
# --------------------------------------------------------------------------- #
def _generate_b_matrix(d):
    vectors = np.zeros((d - 1, d))
    for k in range(1, d):
        v = np.zeros(d)
        v[:k] = 1
        v[k] = -k
        vectors[k - 1] = v / np.sqrt(np.sum(v ** 2))
    return vectors.T


def _generate_aa_matrix(props):
    props = np.asarray(props, dtype=float)
    mat = -np.sqrt(np.outer(props, props))
    np.fill_diagonal(mat, 1 - props)
    return mat


def _compute_s_scores(sample_list, weights, n_all, rng):
    scores = np.empty(len(sample_list))
    for i, x in enumerate(sample_list):
        y = np.concatenate([s for j, s in enumerate(sample_list) if j != i])
        scores[i] = np.sum(_weights_at(weights, rank_x(x, y, rng=rng))) / np.sqrt(n_all)
    return scores


def _gene_msg(msg, gene_name):
    return msg if gene_name is None else f"{msg} for gene {gene_name}"


def _get_h_new(h_w, b_matrix, aa_matrix, gene_name):
    h_mod = b_matrix.T @ aa_matrix @ b_matrix
    h_new = h_w * (h_mod + h_mod.T) / 2
    if not np.all(np.isfinite(h_new)):
        warnings.warn(_gene_msg("Covariance matrix contains NA or Inf values", gene_name)
                      + ", returning NA.", stacklevel=3)
        return None
    return h_new


def _get_h_inv(h_new, gene_name):
    if h_new is None:
        return None
    try:
        if np.allclose(h_new, h_new.T, rtol=0, atol=100 * np.finfo(float).eps) \
                and np.all(np.linalg.eigvalsh(h_new) > 0):
            c = np.linalg.cholesky(h_new)
            c_inv = np.linalg.inv(c)
            return c_inv.T @ c_inv
        return np.linalg.inv(h_new)
    except np.linalg.LinAlgError:
        warnings.warn(_gene_msg("Covariance matrix is not invertible", gene_name)
                      + ", returning NA.", stacklevel=3)
        return None
