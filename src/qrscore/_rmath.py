"""Accurate negative binomial log-density, following R's nmath ``dnbinom_mu``.

R computes NB densities with Loader's saddle-point method (``dbinom_raw``,
``bd0``, ``stirlerr``), which stays accurate when the size parameter is huge.
The naive ``lgamma(y + theta) - lgamma(theta)`` form loses all precision there,
and optimisers that probe large theta can then accept nonsense steps.
"""

from __future__ import annotations

import numpy as np
from scipy import special

_LN_SQRT_2PI = 0.918938533204672741780329736406
_LN_2PI = 1.837877066409345483560659472811
_S0, _S1, _S2, _S3, _S4 = 1 / 12, 1 / 360, 1 / 1260, 1 / 1680, 1 / 1188


def _stirlerr(n):
    """log(n!) - log(sqrt(2*pi*n) * (n/e)^n)."""
    n = np.asarray(n, dtype=float)
    out = np.empty_like(n)
    small = n <= 15
    if np.any(small):
        ns = n[small]
        out[small] = special.gammaln(ns + 1) - (ns + 0.5) * np.log(ns) + ns - _LN_SQRT_2PI
    big = ~small
    if np.any(big):
        nb = n[big]
        nn = nb * nb
        out[big] = np.select(
            [nb > 500, nb > 80, nb > 35],
            [(_S0 - _S1 / nn) / nb,
             (_S0 - (_S1 - _S2 / nn) / nn) / nb,
             (_S0 - (_S1 - (_S2 - _S3 / nn) / nn) / nn) / nb],
            (_S0 - (_S1 - (_S2 - (_S3 - _S4 / nn) / nn) / nn) / nn) / nb)
    return out


def _bd0(x, np_):
    """Deviance term x*log(x/np) + np - x, computed stably."""
    x = np.asarray(x, dtype=float)
    np_ = np.asarray(np_, dtype=float)
    x, np_ = np.broadcast_arrays(x, np_)
    with np.errstate(all="ignore"):
        out = x * np.log(x / np_) + np_ - x
    close = np.abs(x - np_) < 0.1 * (x + np_)
    if np.any(close):
        xc, nc = x[close], np_[close]
        v = (xc - nc) / (xc + nc)
        s = (xc - nc) * v
        ej = 2 * xc * v
        v2 = v * v
        done = np.abs(s) < np.finfo(float).tiny
        for j in range(1, 1000):
            ej = ej * v2
            s1 = s + ej / (2 * j + 1)
            newly = (s1 == s) & ~done
            s = np.where(done, s, s1)
            done |= newly
            if done.all():
                break
        out = out.copy()
        out[close] = s
    return out


def _dbinom_raw_log(x, n, p, q):
    """log dbinom_raw(x; n, p, q) for 0 < x < n (the case dnbinom_mu needs)."""
    lc = _stirlerr(n) - _stirlerr(x) - _stirlerr(n - x) - _bd0(x, n * p) - _bd0(n - x, n * q)
    lf = _LN_2PI + np.log(x) + np.log1p(-x / n)
    return lc - 0.5 * lf


def nb_logpmf(y, mu, size):
    """log dnbinom(y, size = size, mu = mu), vectorised, as R's ``dnbinom_mu``."""
    y = np.asarray(y, dtype=float)
    y, mu, size = np.broadcast_arrays(y, np.asarray(mu, float), np.asarray(size, float))
    out = np.empty(y.shape)
    with np.errstate(all="ignore"):
        z = y == 0
        if np.any(z):
            s, m = size[z], mu[z]
            out[z] = s * np.where(s < m, np.log(s / (s + m)), np.log1p(-m / (s + m)))
        tiny = ~z & (y < 1e-10 * size)
        if np.any(tiny):
            yy, s, m = y[tiny], size[tiny], mu[tiny]
            lp = np.where(s < m, np.log(s / (1 + s / m)), np.log(m / (1 + m / s)))
            out[tiny] = yy * lp - m - special.gammaln(yy + 1) + np.log1p(yy * (yy - 1) / (2 * s))
        # For moderate size the lgamma form is accurate (~1e-12 per term) and much
        # faster; Loader's saddle-point form is only needed for large size.
        simple = ~z & ~tiny & (size <= 1e3)
        if np.any(simple):
            yy, s, m = y[simple], size[simple], mu[simple]
            out[simple] = (special.gammaln(yy + s) - special.gammaln(s) - special.gammaln(yy + 1)
                           + s * np.log(s / (s + m)) + yy * np.log(m / (s + m)))
        rest = ~z & ~tiny & ~simple
        if np.any(rest):
            yy, s, m = y[rest], size[rest], mu[rest]
            ans = _dbinom_raw_log(s, yy + s, s / (s + m), m / (s + m))
            out[rest] = np.log(s / (s + yy)) + ans
    return out
