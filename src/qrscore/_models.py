"""Intercept-only NB and ZINB maximum likelihood fits.

These replace ``MASS::glm.nb(y ~ 1)`` / ``glm.nb(y ~ offset(o))`` and
``pscl::zeroinfl(y ~ 1 | 1, dist = "negbin")`` (optionally with an offset in
the count part), which is all QRscore uses from those packages.
Log-likelihoods are the full likelihoods (including the ``lgamma`` constants),
as returned by ``logLik()`` in R, so likelihood ratio tests match.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import optimize, special

from ._rmath import nb_logpmf

logger = logging.getLogger("qrscore")

# Bounds on log(theta). Very large theta means the data are not overdispersed;
# QRscore caps theta at 100 anyway, so the upper bound only needs to be "big".
_LOG_THETA_MIN, _LOG_THETA_MAX = -15.0, 20.0


class ModelFitError(RuntimeError):
    """Raised when an NB / ZINB fit does not produce finite estimates."""


# Largest NB dispersion used for the weights (R: .BETA_CAP).
BETA_CAP = 1e6


@dataclass
class NBFit:
    mu: float          # exp(intercept): mean at offset 0
    theta: float
    loglik: float
    normalized: bool = False   # True if the offset fit failed and normalised counts were used


@dataclass
class ZINBFit:
    mu: float          # exp(count intercept)
    theta: float
    pi: float          # plogis(zero intercept)
    loglik: float


def _nb_logpmf(y, mu, theta):
    """log NB(y; mu, theta) computed as R's dnbinom (accurate for huge theta)."""
    return nb_logpmf(y, mu, theta)


def _prep(counts, offset):
    y = np.asarray(counts, dtype=float)
    o = np.zeros_like(y) if offset is None else np.asarray(offset, dtype=float)
    if y.shape != o.shape:
        raise ValueError("offset must have the same length as counts")
    if np.any(y < 0) or not np.all(np.isfinite(y)):
        raise ModelFitError("counts must be finite and non-negative")
    if np.all(y == 0):
        raise ModelFitError("all counts are zero; the model is not identifiable")
    return y, o


def _moment_theta(y, mu_i):
    m = np.mean(mu_i)
    v = np.var(y, ddof=1) if y.size > 1 else 0.0
    excess = v - m
    return float(np.clip(m * m / excess, 1e-3, 1e6)) if excess > 1e-8 else 1e4


def fit_nb_mle(counts, offset=None) -> NBFit:
    """MLE of an intercept-only NB model, as ``MASS::glm.nb``."""
    y, o = _prep(counts, offset)
    eo = np.exp(o)

    def negll(par):
        mu_i = np.exp(par[0] + o)
        return -np.sum(_nb_logpmf(y, mu_i, np.exp(par[1])))

    def grad(par):
        b, lt = par
        th = np.exp(lt)
        mu_i = np.exp(b + o)
        d_b = np.sum((y - mu_i) * th / (th + mu_i))
        d_th = np.sum(special.digamma(y + th) - special.digamma(th)
                      + np.log(th) + 1 - np.log(th + mu_i) - (y + th) / (th + mu_i))
        return -np.array([d_b, d_th * th])

    b0 = np.log(np.sum(y) / np.sum(eo))
    lt0 = np.log(_moment_theta(y, np.exp(b0 + o)))
    lt0 = float(np.clip(lt0, _LOG_THETA_MIN + 1, _LOG_THETA_MAX - 1))
    best = None
    for start in ([b0, lt0], [b0, 0.0], [b0, 3.0]):
        res = optimize.minimize(negll, np.array(start), jac=grad, method="L-BFGS-B",
                                bounds=[(None, None), (_LOG_THETA_MIN, _LOG_THETA_MAX)],
                                options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 2000})
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None or not np.all(np.isfinite(best.x)):
        raise ModelFitError("NB fit failed")
    b, lt = best.x
    if offset is None:
        b = b0  # intercept-only NB MLE of the mean is exactly log(mean(y))
        lt = optimize.minimize_scalar(
            lambda t: negll([b, t]), bounds=(_LOG_THETA_MIN, _LOG_THETA_MAX),
            method="bounded", options={"xatol": 1e-12}).x
        if negll([b, lt]) > best.fun + 1e-9:   # keep whichever is better
            b, lt = best.x
    return NBFit(mu=float(np.exp(b)), theta=float(np.exp(lt)), loglik=float(-negll([b, lt])))


def _theta_ml(y, mu, limit=25, eps=np.finfo(float).eps ** 0.25):
    """Port of ``MASS::theta.ml`` (Newton–Raphson from the moment estimate).

    Kept deliberately identical to MASS, including its failure mode: when the
    start ``n / sum((y/mu - 1)^2)`` is far from the optimum the iteration can
    run away to very large theta.
    """
    n = y.size

    def score(th):
        return np.sum(special.digamma(th + y) - special.digamma(th) + np.log(th) + 1
                      - np.log(th + mu) - (y + th) / (mu + th))

    def info(th):
        return np.sum(-special.polygamma(1, th + y) + special.polygamma(1, th) - 1 / th
                      + 2 / (mu + th) - (y + th) / (mu + th) ** 2)

    with np.errstate(all="ignore"):
        t0 = n / np.sum((y / mu - 1) ** 2)
        it, delta = 0, 1.0
        while True:
            it += 1
            if not (it < limit and abs(delta) > eps):
                break
            t0 = abs(t0)
            delta = score(t0) / info(t0)
            t0 = t0 + delta
    if t0 < 0:
        logger.warning("theta.ml: estimate truncated at zero")
        t0 = 0.0
    if it == limit:
        logger.debug("theta.ml: iteration limit reached")
    return float(t0)


def _nb_loglik_mass(th, mu, y):
    """``loglik`` helper inside ``MASS::glm.nb``."""
    with np.errstate(all="ignore"):
        return float(np.sum(special.gammaln(th + y) - special.gammaln(th) - special.gammaln(y + 1)
                            + th * np.log(th) + y * np.log(mu + (y == 0))
                            - (th + y) * np.log(th + mu)))


def _irls_intercept(y, o, theta, b_start, maxit=25, epsilon=1e-8):
    """Intercept-only log-link GLM fit with offset (Poisson if theta is None)."""
    b = b_start
    for _ in range(maxit):
        mu = np.exp(b + o)
        var_w = mu if theta is None else mu / (1 + mu / theta)       # IRLS weight mu^2/V(mu)
        b_new = b + np.sum(var_w * (y - mu) / mu) / np.sum(var_w)
        if abs(b_new - b) < epsilon * (abs(b) + 0.1):
            b = b_new
            break
        b = b_new
    return b


def fit_nb_mass(counts, offset=None, maxit=25, epsilon=1e-8) -> NBFit:
    """Replicates ``MASS::glm.nb(y ~ 1)`` / ``glm.nb(y ~ offset(o))`` step by step."""
    y, o = _prep(counts, offset)
    b = np.log(np.sum(y) / np.sum(np.exp(o)))
    b = _irls_intercept(y, o, None, b)                     # Poisson start
    mu = np.exp(b + o)
    th = _theta_ml(y, mu, limit=maxit)
    d1 = np.sqrt(2 * max(1, y.size - 1))
    d2 = delta = 1.0
    lm = _nb_loglik_mass(th, mu, y)
    lm0 = lm + 2 * d1
    it = 0
    while True:
        it += 1
        if not (it <= maxit and (abs(lm0 - lm) / d1 + abs(delta) / d2) > epsilon):
            break
        if not np.isfinite(th) or th <= 0:
            raise ModelFitError("glm.nb: theta is not finite")
        b = _irls_intercept(y, o, th, b, maxit=maxit, epsilon=epsilon)
        t0 = th
        th = _theta_ml(y, mu, limit=maxit)                  # uses mu from previous fit, as MASS does
        mu = np.exp(b + o)
        delta = t0 - th
        lm0 = lm
        lm = _nb_loglik_mass(th, mu, y)
    return NBFit(mu=float(np.exp(b)), theta=float(th), loglik=lm)


def fit_zinb_mle(counts, offset=None) -> ZINBFit:
    """Multi-start MLE of ``zeroinfl(y ~ offset(o) | 1, dist = "negbin")``."""
    y, o = _prep(counts, offset)
    is0 = y == 0
    if not np.any(is0):
        raise ModelFitError("no zeros in data; ZINB zero component is not identifiable")

    def negll(par):
        b, g, lt = par
        th = np.exp(lt)
        mu_i = np.exp(b + o)
        log_nb = _nb_logpmf(y, mu_i, th)
        log_pi = -np.logaddexp(0.0, -g)        # log plogis(g)
        log_1mpi = -np.logaddexp(0.0, g)       # log(1 - plogis(g))
        ll = np.where(is0, np.logaddexp(log_pi, log_1mpi + log_nb), log_1mpi + log_nb)
        return -np.sum(ll)

    nb = fit_nb_mle(y, None if offset is None else o)
    b0 = np.log(nb.mu)
    lt0 = float(np.clip(np.log(nb.theta), _LOG_THETA_MIN + 1, _LOG_THETA_MAX - 1))
    p0 = np.clip(is0.mean() / 2, 1e-3, 0.9)
    starts = [
        [b0, np.log(p0 / (1 - p0)), lt0],
        [b0, -3.0, lt0],
        [np.log(np.mean(y[~is0]) / np.mean(np.exp(o[~is0]))) if np.any(~is0) else b0,
         special.logit(np.clip(is0.mean(), 1e-3, 0.99)), 1.0],
        [b0, 0.0, 0.0],
    ]
    best = None
    for start in starts:
        res = optimize.minimize(negll, np.array(start, dtype=float), method="L-BFGS-B",
                                bounds=[(None, None), (-30, 30), (_LOG_THETA_MIN, _LOG_THETA_MAX)],
                                options={"ftol": 1e-15, "gtol": 1e-9, "maxiter": 5000})
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise ModelFitError("ZINB fit failed")
    # Polish with Nelder-Mead from the best point (cheap, 3 parameters).
    polish = optimize.minimize(negll, best.x, method="Nelder-Mead",
                               options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000})
    if np.isfinite(polish.fun) and polish.fun < best.fun:
        best = polish
    b, g, lt = best.x
    if not np.all(np.isfinite(best.x)):
        raise ModelFitError("ZINB fit failed")
    return ZINBFit(mu=float(np.exp(b)), theta=float(np.exp(lt)),
                   pi=float(special.expit(g)), loglik=float(-best.fun))


def _dnbinom0_log(theta, mu):
    """log dnbinom(0, size = theta, mu = mu)."""
    return theta * np.where(theta < mu, np.log(theta / (theta + mu)), np.log1p(-mu / (theta + mu)))


def fit_zinb_pscl(counts, offset=None) -> ZINBFit:
    """Replicates ``pscl::zeroinfl(y ~ offset(o) | 1, dist = "negbin")`` (pscl 1.5.x).

    Same starting values (a truncated-count pre-fit, then Poisson / binomial
    intercepts with theta = 1) and the same optimiser (R's BFGS, reltol
    eps^(1/1.6), maxit 10000), so the fit lands where pscl's does.
    """
    from ._optim import vmmin

    y, o = _prep(counts, offset)
    y0, y1 = y <= 0, y > 0
    if not np.any(y0):
        raise ModelFitError("no zeros in data; ZINB zero component is not identifiable")
    reltol, maxit = np.finfo(float).eps ** (1 / 1.6), 10000

    def loglik(par, trunc_start=False):
        mu = np.exp(par[0] + o)
        phi = np.zeros_like(mu) if trunc_start else np.full_like(mu, special.expit(par[1]))
        theta = np.exp(par[2])
        with np.errstate(all="ignore"):
            ll0 = np.log(phi + np.exp(np.log(1 - phi) + _dnbinom0_log(theta, mu)))
            ll1 = np.log(1 - phi) + _nb_logpmf(y, mu, theta)
            if trunc_start:
                return float(np.sum(ll1[y1]) - np.sum(np.log(1 - np.exp(ll0[y1]))))
            return float(np.sum(ll0[y0]) + np.sum(ll1[y1]))

    def grad(par):
        eta = par[0] + o
        mu = np.exp(eta)
        muz = special.expit(par[1])
        mu_eta = muz * (1 - muz)
        theta = np.exp(par[2])
        with np.errstate(all="ignore"):
            cl0 = _dnbinom0_log(theta, mu)
            dens0 = muz * (1 - y1) + np.exp(np.log(1 - muz) + cl0)
            w_count = np.where(y1, y - mu * (y + theta) / (mu + theta),
                               -np.exp(-np.log(dens0) + np.log(1 - muz) + cl0 + np.log(theta)
                                       - np.log(mu + theta) + np.log(mu)))
            w_zero = np.where(y1, -1 / (1 - muz) * mu_eta, (mu_eta - np.exp(cl0) * mu_eta) / dens0)
            w_theta = theta * np.where(
                y1, special.digamma(y + theta) - special.digamma(theta) + np.log(theta)
                - np.log(mu + theta) + 1 - (y + theta) / (mu + theta),
                np.exp(-np.log(dens0) + np.log(1 - muz) + cl0)
                * (np.log(theta) - np.log(mu + theta) + 1 - theta / (mu + theta)))
        return np.array([w_count.sum(), w_zero.sum(), w_theta.sum()])

    def count_grad(par):
        eta = par[0] + o[y1]
        mu = np.exp(eta)
        theta = np.exp(par[1])
        yy = y[y1]
        with np.errstate(all="ignore"):
            lp0 = _dnbinom0_log(theta, mu)                         # log P(Y = 0)
            logratio = lp0 - np.log(-np.expm1(lp0))                # log P(0) - log P(Y > 0)
            r1 = np.sum((yy - mu * (yy + theta) / (mu + theta))
                        - np.exp(logratio + np.log(theta) - np.log(mu + theta) + eta))
            r2 = np.sum(special.digamma(yy + theta) - special.digamma(theta) + np.log(theta)
                        - np.log(mu + theta) + 1 - (yy + theta) / (mu + theta)
                        + np.exp(logratio) * (np.log(theta) - np.log(mu + theta) + 1
                                              - theta / (mu + theta))) * theta
        return np.array([r1, r2])

    # Starting values, exactly as zeroinfl() builds them when EM = FALSE.
    zero_start = float(special.logit(np.mean(y0)))
    lmstart = float(np.mean(np.log(y[y1]) - o[y1]))
    try:
        tpar, _, tconv = vmmin(lambda p: -loglik([p[0], 0.0, p[1]], trunc_start=True),
                               lambda p: -count_grad(p), [lmstart, 0.0], maxit, reltol)
    except FloatingPointError:
        tconv = 1
        tpar = np.array([lmstart, 0.0])
    if tconv == 0:
        start = [float(np.log(np.sum(y) / np.sum(np.exp(o)))), zero_start, 0.0]
    else:
        start = [float(tpar[0]), zero_start, float(tpar[1])]

    par, fval, conv = vmmin(lambda p: -loglik(p), lambda p: -grad(p), start, maxit, reltol)
    if conv > 0:
        logger.warning("zeroinfl: optimization failed to converge")
    if not np.all(np.isfinite(par)):
        raise ModelFitError("ZINB fit failed")
    return ZINBFit(mu=float(np.exp(par[0])), theta=float(np.exp(par[2])),
                   pi=float(special.expit(par[1])), loglik=-float(fval))


_FIT_METHODS = ("r", "mle")


def _check_method(method):
    if method not in _FIT_METHODS:
        raise ValueError("fit_method must be 'r' or 'mle'")


def fit_nb(counts, offset=None, method="r") -> NBFit:
    """Fit an intercept-only NB model (optional log size-factor offset).

    method="r" (default) reproduces ``MASS::glm.nb`` step by step, including its
    occasional divergence of theta; method="mle" returns the true maximum.
    """
    _check_method(method)
    return fit_nb_mass(counts, offset) if method == "r" else fit_nb_mle(counts, offset)


def fit_zinb(counts, offset=None, method="r") -> ZINBFit:
    """Fit an intercept-only ZINB model (optional offset in the count part).

    method="r" (default) reproduces ``pscl::zeroinfl`` (same start values and
    BFGS optimiser); method="mle" uses a multi-start search for the global maximum.
    """
    _check_method(method)
    return fit_zinb_pscl(counts, offset) if method == "r" else fit_zinb_mle(counts, offset)


def _nb_mle_r(counts, offset=None):
    """Port of QRscore's ``.nb_mle``: ``optim(BFGS)`` on (log mu, log theta) with R's
    finite-difference gradient (ndeps = 1e-3), reltol 1e-12, from two starts."""
    from ._optim import vmmin

    y = np.asarray(counts, dtype=float)
    o = np.zeros_like(y) if offset is None else np.asarray(offset, dtype=float)

    def nll(par):
        with np.errstate(all="ignore"):
            return float(-np.sum(nb_logpmf(y, np.exp(par[0] + o), np.exp(par[1]))))

    def num_grad(par, h=1e-3):
        g = np.empty(2)
        for i in range(2):
            up, dn = par.copy(), par.copy()
            up[i] += h
            dn[i] -= h
            g[i] = (nll(up) - nll(dn)) / (2 * h)
            if not np.isfinite(g[i]):
                raise FloatingPointError("non-finite finite-difference value")
        return g

    b0 = float(np.log(np.sum(y) / np.sum(np.exp(o))))
    best = None
    for lt0 in (0.0, float(np.log(10))):
        try:
            par, val, _ = vmmin(nll, num_grad, [b0, lt0], maxit=1000, reltol=1e-12)
        except FloatingPointError:
            continue
        if np.isfinite(val) and (best is None or val < best[1]):
            best = (par, val)
    if best is None:
        return None
    return NBFit(mu=float(np.exp(best[0][0])), theta=float(np.exp(best[0][1])), loglik=-float(best[1]))


def fit_nb_model(counts, offset=None, method="r") -> NBFit:
    """NB fit used by the tests (R: ``.fit_nb_model``).

    method="r": ``MASS::glm.nb`` replica, replaced by a direct maximisation when
    that reaches a higher likelihood (glm.nb's theta.ml can diverge).
    method="mle": multi-start maximum likelihood.
    If the offset model fails, normalised counts are fitted without offset and
    the result is flagged ``normalized``.
    """
    _check_method(method)
    base = fit_nb_mass if method == "r" else fit_nb_mle
    try:
        res = base(counts, offset)
    except (ModelFitError, FloatingPointError, ValueError):
        if offset is None:
            raise
        logger.info("Offset model failed, using normalized counts.")
        res = fit_nb_model(np.round(np.asarray(counts, float) / np.exp(offset)), None, method)
        res.normalized = True
        return res
    if method == "r":
        mle = _nb_mle_r(counts, offset)
        if mle is not None and (not np.isfinite(res.loglik) or mle.loglik > res.loglik + 1e-6):
            logger.info("glm.nb did not reach the NB maximum likelihood estimate; "
                        "using direct maximisation.")
            res = mle
    return res
