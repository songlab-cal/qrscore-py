"""Regression against outputs of the R package (QRscore 1.5.2).

``r_reference/r_results.csv`` holds outputs of the R package on the same
inputs with QRscore 1.5.2 in R. Both sides use the same deterministic tie-breaking in ``rank_x`` so the
comparison is exact up to optimiser tolerance; with R's random tie-breaking the
p-values of tied count data vary from run to run in R too.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import qrscore as q
import qrscore._utils as U
import qrscore.core as core

REF = Path(__file__).parent / "r_reference"


def _det_rank(x, y=None, ties_break=True, rng=None):
    x = np.asarray(x, float)
    y = np.empty(0) if y is None else np.asarray(y, float)
    d = np.diff(np.sort(np.concatenate([x, y])))
    g = d[d > 0].min() if np.any(d > 0) else 1
    c = np.sort(np.concatenate([x, y + g / 4]))
    return np.searchsorted(c, np.sort(x), "left").astype(float) + 1


@pytest.fixture(scope="module")
def results():
    mp = pytest.MonkeyPatch()
    mp.setattr(core, "rank_x", _det_rank)
    mp.setattr(U, "rank_x", _det_rank)
    res = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for pr in [(2, 10, .1, 50, 25), (0.5, 200, 0, 300, 120), (5, 40, .1, 100, 60),
                   (100, 3, .3, 20, 9), (1.3, 1500, .05, 80, 40), (1e4, 5, .1, 60, 40),
                   (1e6, 1000, 0, 60, 40), (1e6, 3, .2, 60, 40), (100, 9000, 0, 60, 40)]:
            tag = "_".join(f"{v:g}" for v in pr)
            for name, fn in (("wmean", q.computeweight_mean), ("wdisp", q.computeweight_disp)):
                w = fn(*pr)
                for i, v in enumerate(w["weight"], 1):
                    res[f"{name}_w_{tag}_{i}"] = v
                res[f"{name}_v_{tag}"] = w["var"]
        for alt in ("two.sided", "greater", "less"):
            res[f"comp_exact_{alt}"] = q.get_composition_pvalue(0.31, 25, 6, 1, np.arange(5, -1, -1) / 5, alt)
            res[f"comp_exact2_{alt}"] = q.get_composition_pvalue(0.2, 30, 5, 2, [1, .2, 0, .2, 1], alt)
        xa, ya = pd.read_csv(REF / "flex_x.csv").v.values, pd.read_csv(REF / "flex_y.csv").v.values
        for p in (1, 2):
            for alt in ("two.sided", "greater", "less"):
                res[f"flex_asym_p{p}_{alt}"] = q.qrscore_flex(
                    xa[:60], ya, p, np.arange(60, -1, -1.0) ** p, alt, "asymptotic")
        mw = pd.read_csv(REF / "mw.csv").v.values
        for alt in ("two.sided", "greater", "less"):
            res[f"mw_exact_{alt}"] = q.qrscore_test(mw[:45], np.repeat(["A", "B"], [20, 25]), p=1,
                                                    w_list=np.arange(20, -1, -1), alternative=alt)
            res[f"mw_ties_{alt}"] = q.qrscore_test(mw[45:], np.repeat(["A", "B"], [70, 80]), p=1,
                                                   w_list=np.arange(70, -1, -1), alternative=alt)
        for i in range(1, 17):
            d = pd.read_csv(REF / f"z{i:02d}.csv")
            x, y, sf = d.v[d.g == "A"].values, d.v[d.g == "B"].values, d.sf.values
            for meas in ("mean", "dispersion"):
                for zi in (True, False):
                    for lr in (False, True):
                        tag = f"z{i:02d}_{meas}_zi{int(zi)}_lr{int(lr)}"
                        kw = dict(zero_inflation=zi, lr_test=lr, measure=meas, rng=1)
                        res[tag + "_asym"] = q.qrscore_zinb(x, y, approx="asymptotic", **kw)
                        res[tag + "_asym_sf"] = q.qrscore_zinb(x, y, size_factors=sf, approx="asymptotic", **kw)
                        if x.size < 10 and y.size < 10:
                            res[tag + "_exact"] = q.qrscore_zinb(x, y, approx="resample", **kw)
        for i in range(1, 9):
            d = pd.read_csv(REF / f"k{i:02d}.csv")
            for meas in ("mean", "dispersion"):
                for zi in (True, False):
                    tag = f"k{i:02d}_{meas}_zi{int(zi)}"
                    kw = dict(approx="asymptotic", zero_inflation=zi, measure=meas, rng=1)
                    res[tag] = q.qrscore_test(d.v.values, d.g.values, **kw)
                    res[tag + "_sf"] = q.qrscore_test(d.v.values, d.g.values, size_factors=d.sf.values, **kw)
        for i in range(1, 13):        # datasets where MASS::glm.nb's theta.ml diverges
            d = pd.read_csv(REF / f"d{i:02d}.csv")
            y, h = d.y.values.astype(float), len(d) // 2
            for meas in ("mean", "dispersion"):
                for lr in (False, True):
                    tag = f"nbdiv{i:02d}_{meas}_lr{int(lr)}"
                    kw = dict(zero_inflation=lr, lr_test=lr, approx="asymptotic", measure=meas, rng=1)
                    res[tag] = q.qrscore_zinb(y[:h], y[h:], **kw)
                    res[tag + "_sf"] = q.qrscore_zinb(y[:h], y[h:], size_factors=np.exp(d.o.values), **kw)
        for i in range(1, 7):         # first label is the larger group, with size factors
            d = pd.read_csv(REF / f"z{i + 6:02d}.csv")
            for meas in ("mean", "dispersion"):
                res[f"swap{i:02d}_{meas}"] = q.qrscore_test(
                    d.v.values, np.where(d.g == "A", "Z", "A"), size_factors=d.sf.values,
                    measure=meas, zero_inflation=False, approx="asymptotic", rng=1)
    mp.undo()
    r = pd.read_csv(REF / "r_results.csv").set_index("case").r
    return r, pd.Series(res)[r.index]


def test_same_cases_and_missingness(results):
    r, py = results
    assert (r.isna() == py.isna()).all()


def test_weights_match_r(results):
    r, py = results
    w = r.index.str.startswith("w")
    np.testing.assert_allclose(py[w], r[w], rtol=1e-9)


def test_pvalues_match_r(results):
    r, py = results
    pv = ~r.index.str.startswith("w") & r.notna()
    # Deterministic paths agree to ~1e-12; ZINB fits on flat likelihoods can stop a
    # hair apart from pscl's (last-bit floating point), moving p by < 1e-3.
    np.testing.assert_allclose(py[pv], r[pv], rtol=5e-3, atol=1e-3)
    # Without ZINB fits and without exact enumeration (whose null distribution has
    # atoms exactly at the observed statistic, so "<" vs "==" is decided by rounding
    # order in either language), results agree to optimiser precision.
    strict = pv & ~r.index.str.contains("zi1") & ~r.index.str.contains("exact")
    np.testing.assert_allclose(py[strict], r[strict], rtol=1e-5, atol=1e-9)
