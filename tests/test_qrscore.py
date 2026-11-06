"""Ports of the R package's testthat suite (tests/testthat/*.R)."""

import warnings

import numpy as np
import pandas as pd
import pytest

import qrscore as q
from qrscore._distributions import dzinbinom, qzinbinom, qnbinom


# ---- test_QRscore.R -------------------------------------------------------
def test_flex_one_sample():
    rng = np.random.default_rng(1)
    p = q.qrscore_flex(np.abs(rng.normal(size=10)), p=2, w_list=np.ones(10), rng=rng)
    assert 0 <= p <= 1


def test_flex_two_sample():
    rng = np.random.default_rng(1)
    p = q.qrscore_flex(np.abs(rng.normal(size=30)), np.abs(rng.normal(size=100)), p=2,
                       w_list=np.ones(31), resamp_number=5000, rng=rng)
    assert 0 <= p <= 1


def test_flex_two_sample_asymptotic():
    rng = np.random.default_rng(1)
    p = q.qrscore_flex(np.abs(rng.normal(size=100)), np.abs(rng.normal(size=100)), p=1,
                       w_list=np.arange(101), approx="asymptotic")
    assert 0 <= p <= 1


def test_zinb_runs():
    x = q.rzinbinom(30, size=2, mu=20, pi=0.1, rng=1)
    y = q.rzinbinom(30, size=1, mu=20, pi=0.1, rng=2)
    p = q.qrscore_zinb(x, y, zero_inflation=True, approx="asymptotic", measure="dispersion", rng=3)
    assert 0 <= p <= 1


def test_zinb_calibration_and_power():
    rng = np.random.default_rng(42)
    n_sim = 300
    null, alt = [], []
    for _ in range(n_sim):
        x, y = q.rzinbinom(100, size=2, mu=20, rng=rng), q.rzinbinom(100, size=2, mu=20, rng=rng)
        null.append(q.qrscore_zinb(x, y, zero_inflation=False, approx="asymptotic", measure="mean", rng=rng))
        x, y = q.rzinbinom(100, size=2, mu=20, rng=rng), q.rzinbinom(100, size=10, mu=20, rng=rng)
        alt.append(q.qrscore_zinb(x, y, zero_inflation=False, approx="asymptotic", measure="dispersion", rng=rng))
    assert np.nanmean(np.array(null) < 0.05) <= 0.08
    assert np.nanmean(np.array(alt) < 0.05) > 0.5


# ---- test_auxiliary.R -----------------------------------------------------
def test_dzinbinom_warns_on_bad_pi():
    with pytest.warns(UserWarning):
        dzinbinom(1, mu=10, theta=5, pi=-0.1)
    with pytest.warns(UserWarning):
        dzinbinom(1, mu=10, theta=5, pi=1.1)


def test_dzinbinom_values():
    from scipy import stats
    x = np.arange(6)
    expected = np.log(0.8) + stats.nbinom(n=2, p=2 / 5).logpmf(x)
    expected[0] = np.log(np.exp(expected[0]) + 0.2)
    np.testing.assert_allclose(dzinbinom(x, mu=3, theta=2, pi=0.2, log=True), expected, rtol=1e-12)


def test_qzinbinom():
    with pytest.warns(UserWarning):
        qzinbinom(0.5, mu=10, theta=5, pi=-0.1)
    assert qzinbinom(0.5, mu=10, theta=5, pi=0) == qnbinom(0.5, mu=10, size=5)


def test_rzinbinom():
    with pytest.warns(UserWarning):
        q.rzinbinom(10, mu=10, theta=5, pi=1.1, rng=1)
    s = q.rzinbinom(100, mu=10, theta=5, pi=0.2, rng=1)
    assert s.size == 100 and np.sum(s == 0) > 0


def test_composition_pvalue_exact():
    assert q.get_composition_pvalue(0.5, 30, 5, 1, np.ones(5), "two.sided") >= 0


def test_rank_x():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=5), rng.normal(size=5)
    r = q.rank_x(x, y, rng=rng)
    assert np.all((r >= 1) & (r <= 10)) and np.unique(r).size == r.size


@pytest.mark.parametrize("fn", [q.computeweight_mean, q.computeweight_disp])
def test_computeweight(fn):
    res = fn(beta=2, mu=10, pi=0.1, n=50, k=25)
    assert set(res) == {"weight", "var"}
    assert res["weight"].size == 75 and not np.any(np.isnan(res["weight"]))
    assert np.isfinite(res["var"])


# ---- test_de_analysis.R ---------------------------------------------------
def _nb_matrix(n_genes, n_samples, seed):
    rng = np.random.default_rng(seed)
    m = rng.negative_binomial(2, 2 / 22, size=(n_genes, n_samples))
    return pd.DataFrame(m, index=[f"gene{i + 1}" for i in range(n_genes)])


def test_genetest_two_groups():
    data = _nb_matrix(10, 200, 1)
    labels = np.repeat(["A", "B"], 100)
    res = q.qrscore_genetest(data, labels, test_mean=True, test_dispersion=True,
                             pairwise_test=True, pairwise_logfc=True)
    assert isinstance(res["mean_test"], pd.DataFrame) and isinstance(res["var_test"], pd.DataFrame)
    assert set(res["mean_test"].index) == set(data.index)
    assert list(res["mean_test"].columns[:2]) == ["QRscore_Mean_p_value", "QRscore_Mean_adj_p_value"]


def test_genetest_pairwise_columns():
    data = _nb_matrix(5, 300, 2)
    labels = np.repeat(["X", "Y", "Z"], 100)
    res = q.qrscore_genetest(data, labels, pairwise_test=True, pairwise_logfc=True)
    cols = res["mean_test"].columns
    assert any("Pairwise_Test" in c for c in cols) and any("Log_FC_Mean" in c for c in cols)


def test_genetest_reproducible_across_cores():
    data = _nb_matrix(6, 120, 3)
    labels = np.repeat(["A", "B"], 60)
    kw = dict(pairwise_test=True, test_dispersion=True, seed=7)
    a = q.qrscore_genetest(data, labels, num_cores=1, **kw)
    b = q.qrscore_genetest(data, labels, num_cores=2, **kw)
    pd.testing.assert_frame_equal(a["mean_test"], b["mean_test"])
    pd.testing.assert_frame_equal(a["var_test"], b["var_test"])


def test_qrscoretest_one_sample():
    rng = np.random.default_rng(1)
    p = q.qrscore_test(np.abs(rng.normal(size=10)), p=2, w_list=np.ones(10), rng=rng)
    assert 0 <= p <= 1


def test_qrscoretest_two_sample_weights():
    rng = np.random.default_rng(1)
    data = np.r_[rng.normal(size=50), rng.normal(1, size=50)]
    p = q.qrscore_test(data, np.repeat(["A", "B"], 50), p=1, w_list=np.arange(50, -1, -1))
    assert 0 <= p <= 1


def test_qrscoretest_nb_and_multisample():
    rng = np.random.default_rng(1)
    data = np.r_[q.rzinbinom(100, size=2, mu=20, rng=rng), q.rzinbinom(100, size=2, mu=30, rng=rng)]
    p = q.qrscore_test(data, np.repeat(["A", "B"], 100), approx="asymptotic",
                       zero_inflation=False, measure="mean", rng=rng)
    assert 0 <= p <= 1
    data = np.concatenate([q.rzinbinom(100, size=s, mu=20, rng=rng) for s in (1, 2, 3)])
    p = q.qrscore_test(data, np.repeat(["G1", "G2", "G3"], 100), approx="asymptotic",
                       zero_inflation=False, measure="dispersion", rng=rng)
    assert 0 <= p <= 1


def test_errors_become_nan_with_warning():
    with pytest.warns(UserWarning, match="QRscoreTest failed"):
        assert np.isnan(q.qrscore_test(np.ones(5), alternative="sideways"))


def test_example_dataset():
    d = q.load_example_dataset()
    assert d["example_data1"].shape == (500, 755) and d["example_data2"].shape == (500, 408)
    assert d["labels1"].size == 755 and d["size_factors2"].size == 408


# ---- numerical behaviour (mirrors the R tests) -------------------------------
def test_size_factors_aligned_when_first_group_larger():
    rng = np.random.default_rng(9)
    sf_a, sf_b = rng.uniform(0.3, 3, 120), rng.uniform(0.3, 3, 60)
    a, b = rng.negative_binomial(3, 3 / (3 + 20 * sf_a)), rng.negative_binomial(3, 3 / (3 + 20 * sf_b))
    p_test = q.qrscore_test(np.r_[a, b], np.repeat(["A", "B"], [120, 60]), size_factors=np.r_[sf_a, sf_b],
                            measure="mean", zero_inflation=False, approx="asymptotic", rng=1)
    p_direct = q.qrscore_zinb(b, a, size_factors=np.r_[sf_b, sf_a], measure="mean",
                              zero_inflation=False, approx="asymptotic", rng=1)
    assert p_test == pytest.approx(p_direct)


def test_nb_fit_reaches_mle_when_glm_nb_diverges():
    from qrscore._models import fit_nb_mass, fit_nb_model
    y = pd.read_csv(__import__("pathlib").Path(__file__).parent / "r_reference" / "d01.csv").y.values
    mass, fit = fit_nb_mass(y), fit_nb_model(y)
    assert mass.theta > 1e4                        # glm.nb replica diverges here, as in R
    assert fit.loglik > mass.loglik + 100 and fit.theta < 1


def test_mean_weight_variance_matches_weights():
    for b, mu, pi in [(2, 10, 0), (5, 3, 0.2), (100, 5, 0), (1e4, 3, 0.1)]:
        res = q.computeweight_mean(b, mu, pi, 20000, 20000)
        assert res["var"] == pytest.approx(np.var(res["weight"], ddof=1), rel=0.01)


def test_large_beta_poisson_limit():
    from qrscore._utils import _z_function_disp
    x = np.arange(11)
    for b in (1e3, 1e6):
        assert not np.any(np.isnan(q.computeweight_disp(b, 4, 0.1, 50, 25)["weight"]))
        assert not np.any(np.isnan(q.computeweight_mean(b, 4, 0.1, 50, 25)["weight"]))
    np.testing.assert_allclose(1e8 * _z_function_disp(x, 1e8, 4, 0), -((x - 4) ** 2 - x) / 2, rtol=1e-5)


def test_weights_finite_when_p0_underflows():
    for b, mu, pi in [(1e6, 1000, 0), (1e6, 5000, 0.1), (100, 9000, 0)]:
        d, m = q.computeweight_disp(b, mu, pi, 2000, 2000), q.computeweight_mean(b, mu, pi, 2000, 2000)
        assert np.all(np.isfinite(np.r_[d["weight"], m["weight"], d["var"], m["var"]]))
        assert d["var"] == pytest.approx(np.var(d["weight"], ddof=1), rel=0.05)


def test_weights_equal_zinb_score():
    from scipy import stats
    from qrscore._utils import _nb_zero_share, _z_function_disp
    h = 1e-6

    def loglik(x, mu, size, pi):
        f = stats.nbinom(n=size, p=size / (size + mu))
        return np.log(pi + (1 - pi) * f.pmf(0)) if x == 0 else np.log(1 - pi) + f.logpmf(x)

    for b, mu, pi in [(2, 10, 0), (5, 3, 0.2), (0.5, 40, 0.5)]:
        x = np.arange(11)
        sd = [(loglik(v, mu, b * (1 + h), pi) - loglik(v, mu, b * (1 - h), pi)) / (2 * h) for v in x]
        np.testing.assert_allclose(_z_function_disp(x, b, mu, pi), sd, rtol=1e-5, atol=1e-7)
        sm = [(loglik(v, mu * np.exp(h), b, pi) - loglik(v, mu * np.exp(-h), b, pi)) / (2 * h) for v in x]
        u = mu / (b + mu)
        zm = np.where(x == 0, -mu * (1 - u) * _nb_zero_share(b, np.log1p(-u), pi), (1 - u) * (x - mu))
        np.testing.assert_allclose(zm, sm, rtol=1e-5, atol=1e-7)
