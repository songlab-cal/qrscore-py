"""Port of R's ``optim(method = "BFGS")`` (``vmmin`` in src/appl/optim.c).

Used so that ZINB fits follow exactly the same path as ``pscl::zeroinfl``,
which matters because that optimiser can stop at a different (local) optimum
than a generic optimiser would.
"""

from __future__ import annotations

import math

import numpy as np

_STEPREDN = 0.2
_ACCTOL = 0.0001
_RELTEST = 10.0


def vmmin(fn, gr, x0, maxit=100, reltol=np.finfo(float).eps ** 0.5, abstol=-np.inf):
    """Minimise ``fn`` with R's variable-metric BFGS. Returns (par, value, convergence)."""
    b = np.array(x0, dtype=float)
    n = b.size
    f = fn(b)
    if not math.isfinite(f):
        raise FloatingPointError("initial value in 'vmmin' is not finite")
    fmin = f
    g = np.asarray(gr(b), dtype=float)
    iter_ = 1
    gradcount = 1
    ilast = gradcount
    B = np.eye(n)
    count = 0
    while True:
        if ilast == gradcount:
            B = np.eye(n)
        X = b.copy()
        c = g.copy()
        t = -(B @ g)
        gradproj = float(t @ g)
        if gradproj < 0.0:
            steplength = 1.0
            accpoint = False
            while True:
                b = X + steplength * t
                count = int(np.sum(_RELTEST + X == _RELTEST + b))
                if count < n:
                    f = fn(b)
                    accpoint = math.isfinite(f) and f <= fmin + gradproj * steplength * _ACCTOL
                    if not accpoint:
                        steplength *= _STEPREDN
                if count == n or accpoint:
                    break
            enough = (f > abstol) and abs(f - fmin) > reltol * (abs(fmin) + reltol)
            if not enough:
                count = n
                fmin = f
            if count < n:
                fmin = f
                g = np.asarray(gr(b), dtype=float)
                gradcount += 1
                iter_ += 1
                t = steplength * t
                c = g - c
                d1 = float(t @ c)
                if d1 > 0:
                    Xv = B @ c
                    d2 = 1.0 + float(Xv @ c) / d1
                    B = B + (d2 * np.outer(t, t) - np.outer(Xv, t) - np.outer(t, Xv)) / d1
                else:
                    ilast = gradcount
            else:
                if ilast < gradcount:
                    count = 0
                    ilast = gradcount
        else:
            count = 0
            if ilast == gradcount:
                count = n
            else:
                ilast = gradcount
        if iter_ >= maxit:
            break
        if gradcount - ilast > 2 * n:
            ilast = gradcount
        if count == n and ilast == gradcount:
            break
    return b, fmin, (0 if iter_ < maxit else 1)
