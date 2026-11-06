"""Zero-inflated negative binomial helpers (R: .dzinbinom, .qzinbinom, rzinbinom).

Parameterisation follows R's ``dnbinom(mu=, size=)``: mean ``mu`` and
dispersion ``size`` (also called ``theta`` / ``beta`` in QRscore).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

from ._rmath import nb_logpmf

_DBL_EPS = np.finfo(float).eps


def _resolve_size(theta, size):
    if theta is not None and size is not None:
        raise ValueError("only 'theta' or 'size' may be specified")
    return size if size is not None else theta


def _check_pi(pi):
    pi = np.asarray(pi, dtype=float)
    if np.any(pi < 0) or np.any(pi > 1):
        warnings.warn("'pi' must be in [0, 1]", stacklevel=3)


def _nb(mu, size):
    """scipy NB with R's (mu, size) parameterisation."""
    return stats.nbinom(n=size, p=size / (size + mu))


def qnbinom(p, mu, size):
    """R-compatible ``qnbinom(p, mu=, size=)``.

    R applies a small left-continuity fuzz, ``p * (1 - 64 * DBL_EPSILON)``,
    before searching for the smallest x with F(x) >= p. Reproducing that keeps
    quantiles identical at the boundaries that QRscore's weight grids hit.
    """
    p = np.asarray(p, dtype=float)
    dist = _nb(mu, size)
    p_fuzz = p * (1 - 64 * _DBL_EPS)
    q = dist.ppf(p_fuzz)
    q = np.where(p <= 0, 0.0, q)
    # Guard against off-by-one from scipy's own search.
    q = np.where((q > 0) & (dist.cdf(q - 1) >= p_fuzz), q - 1, q)
    q = np.where(dist.cdf(q) < p_fuzz, q + 1, q)
    return q


def dzinbinom(x, mu, theta=None, size=None, pi=0.0, log=False):
    """Density of the zero-inflated negative binomial (R: ``.dzinbinom``)."""
    _check_pi(pi)
    theta = _resolve_size(theta, size)
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rval = np.asarray(np.log(1 - pi) + nb_logpmf(x, mu, theta), dtype=float)
    x0 = x == 0
    if np.any(x0):
        with np.errstate(divide="ignore"):
            rval = np.where(x0, np.log(np.exp(rval) + pi), rval)
    return rval if log else np.exp(rval)


def qzinbinom(p, mu, theta=None, size=None, pi=0.0, lower_tail=True, log_p=False):
    """Quantile function of the ZINB (R: ``.qzinbinom``)."""
    _check_pi(pi)
    theta = _resolve_size(theta, size)
    p = np.asarray(p, dtype=float)
    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1 - p
    p = np.maximum(0.0, (p - pi) / (1 - pi))
    return qnbinom(p, mu=mu, size=theta)


def rzinbinom(n, mu, theta=None, size=None, pi=0.0, rng=None):
    """Draw ``n`` samples from a ZINB distribution (R: ``rzinbinom``).

    Parameters
    ----------
    n : int
        Number of samples.
    mu : float or array
        Mean of the negative binomial component.
    theta, size : float
        Dispersion (size) parameter; give one of the two.
    pi : float
        Zero-inflation probability in [0, 1].
    rng : numpy.random.Generator, int or None
        Random generator or seed.
    """
    _check_pi(pi)
    theta = _resolve_size(theta, size)
    rng = np.random.default_rng(rng)
    mu = np.broadcast_to(np.asarray(mu, dtype=float), (n,))
    rval = rng.negative_binomial(theta, theta / (theta + mu), size=n).astype(float)
    rval[rng.random(n) < pi] = 0.0
    return rval
