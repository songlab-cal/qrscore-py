"""Gene-wise differential mean / dispersion analysis (R: de_analysis.R)."""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .core import qrscore_test

logger = logging.getLogger("qrscore")


def p_adjust_bh(p):
    """Benjamini–Hochberg adjustment, as R ``p.adjust(p, "BH")`` (NaNs kept, not counted)."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = pv.size
    if m == 0:
        return out
    order = np.argsort(pv)[::-1]
    ranks = np.arange(m, 0, -1)
    adj = np.minimum.accumulate(m / ranks * pv[order])
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def _log2_ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.log2(a / b))


def _add_test_result(x, y, label1, label2, gene_name, measure, do_test, do_logfc,
                     out, suffix, sf_x, sf_y, rng, kwargs):
    if do_test:
        sf = None if sf_x is None else np.concatenate([sf_x, sf_y])
        out[f"QRscore_{suffix}_p_value"] = qrscore_test(
            np.concatenate([x, y]), np.repeat([label1, label2], [x.size, y.size]),
            size_factors=sf, measure=measure, gene_name=gene_name, rng=rng, **kwargs)
    if do_logfc:
        stat = np.mean if measure == "mean" else (lambda v: np.var(v, ddof=1))
        if sf_x is not None:
            x, y = x / sf_x, y / sf_y
        out[f"Log_FC_{suffix}_{label2}_vs_{label1}"] = _log2_ratio(stat(y), stat(x))
    return out


def _pairwise_tests(gene_vec, labels, unique_labels, test_mean, test_dispersion,
                    pairwise_test, pairwise_logfc, gene_name, mean_out, var_out,
                    size_factors, rng, kwargs):
    for i in range(len(unique_labels) - 1):
        for j in range(i + 1, len(unique_labels)):
            li, lj = unique_labels[i], unique_labels[j]
            idx_i, idx_j = labels == li, labels == lj
            x, y = gene_vec[idx_i], gene_vec[idx_j]
            sf_i = None if size_factors is None else size_factors[idx_i]
            sf_j = None if size_factors is None else size_factors[idx_j]
            sf_pair = None if size_factors is None else np.concatenate([sf_i, sf_j])
            lbl = f"{lj}_vs_{li}"
            pair_labels = np.repeat([li, lj], [x.size, y.size])
            if pairwise_test and test_mean:
                mean_out[f"Pairwise_Test_Mean_{lbl}"] = qrscore_test(
                    np.concatenate([x, y]), pair_labels, size_factors=sf_pair, measure="mean",
                    gene_name=gene_name, rng=rng, **kwargs)
            if pairwise_test and test_dispersion:
                var_out[f"Pairwise_Test_Var_{lbl}"] = qrscore_test(
                    np.concatenate([x, y]), pair_labels, size_factors=sf_pair,
                    measure="dispersion", gene_name=gene_name, rng=rng, **kwargs)
            xn, yn = (x, y) if size_factors is None else (x / sf_i, y / sf_j)
            if pairwise_logfc and test_mean:
                mean_out[f"Log_FC_Mean_{lbl}"] = _log2_ratio(np.mean(yn), np.mean(xn))
            if pairwise_logfc and test_dispersion:
                var_out[f"Log_FC_Var_{lbl}"] = _log2_ratio(np.var(yn, ddof=1), np.var(xn, ddof=1))
    return mean_out, var_out


def _one_gene(args):
    (gene_vec, gene_name, labels, size_factors, pairwise_test, pairwise_logfc,
     test_mean, test_dispersion, seed, verbose, kwargs) = args
    if not verbose:
        logging.getLogger("qrscore").setLevel(logging.WARNING)
    rng = np.random.default_rng(seed)
    unique_labels = np.sort(pd.unique(labels))
    mean_out, var_out = {}, {}
    if len(unique_labels) == 2:
        ix, iy = labels == unique_labels[0], labels == unique_labels[1]
        x, y = gene_vec[ix], gene_vec[iy]
        sf_x = None if size_factors is None else size_factors[ix]
        sf_y = None if size_factors is None else size_factors[iy]
        if test_mean:
            _add_test_result(x, y, unique_labels[0], unique_labels[1], gene_name, "mean",
                             pairwise_test, pairwise_logfc, mean_out, "Mean", sf_x, sf_y, rng, kwargs)
        if test_dispersion:
            _add_test_result(x, y, unique_labels[0], unique_labels[1], gene_name, "dispersion",
                             pairwise_test, pairwise_logfc, var_out, "Var", sf_x, sf_y, rng, kwargs)
    elif len(unique_labels) > 2:
        if test_mean:
            mean_out["QRscore_Mean_p_value"] = qrscore_test(
                gene_vec, labels, size_factors=size_factors, measure="mean",
                gene_name=gene_name, rng=rng, **kwargs)
        if test_dispersion:
            var_out["QRscore_Var_p_value"] = qrscore_test(
                gene_vec, labels, size_factors=size_factors, measure="dispersion",
                gene_name=gene_name, rng=rng, **kwargs)
        if pairwise_test or pairwise_logfc:
            mean_out, var_out = _pairwise_tests(
                gene_vec, labels, unique_labels, test_mean, test_dispersion, pairwise_test,
                pairwise_logfc, gene_name, mean_out, var_out, size_factors, rng, kwargs)
    return mean_out, var_out


def _format(results, gene_names, test_type, pval_name, rank):
    rows = [(g, r) for g, r in zip(gene_names, results) if r]
    if not rows:
        return None
    df = pd.DataFrame([r for _, r in rows], index=pd.Index([g for g, _ in rows], name="gene"))
    adj = f"{test_type}_adj_p_value"
    if pval_name not in df.columns:
        return df
    df[adj] = p_adjust_bh(df[pval_name].to_numpy())
    if rank:
        df = df.sort_values(pval_name, kind="stable", na_position="last")
    front = [pval_name, adj]
    return df[front + [c for c in df.columns if c not in front]]


def qrscore_genetest(expr_matrix, labels, size_factors=None, pairwise_test=False,
                     pairwise_logfc=False, test_mean=True, test_dispersion=False,
                     num_cores=1, verbose=False, rank=True, seed=1, **kwargs):
    """Differential mean / dispersion analysis across genes (R: ``QRscoreGenetest``).

    Parameters
    ----------
    expr_matrix : pandas.DataFrame or 2-D array
        Raw counts, genes x samples. A DataFrame's index gives gene names.
    labels : array-like
        Group label of each sample (column).
    size_factors : array-like, optional
        Per-sample size factors (e.g. from DESeq2 / pydeseq2); must be > 0.
    pairwise_test, pairwise_logfc : bool
        For more than two groups, also run / report all pairwise comparisons.
    test_mean, test_dispersion : bool
        Which shifts to test.
    num_cores : int
        Worker processes. Results do not depend on it: each gene gets its own
        random stream derived from ``seed``.
    verbose : bool
        If False, informational log messages are silenced.
    rank : bool
        Sort each table by its main p-value.
    seed : int
        Base seed.
    **kwargs
        Passed to :func:`qrscore_test` (e.g. ``approx="asymptotic"``,
        ``zero_inflation=False``, ``fit_method="mle"``).

    Returns
    -------
    dict with keys "mean_test" and "var_test" (pandas DataFrames or None).
    """
    if isinstance(expr_matrix, pd.DataFrame):
        gene_names = list(expr_matrix.index.astype(str))
        mat = expr_matrix.to_numpy(dtype=float)
    else:
        mat = np.asarray(expr_matrix, dtype=float)
        gene_names = [f"gene{i + 1}" for i in range(mat.shape[0])]
    labels = np.asarray(labels)
    if labels.size != mat.shape[1]:
        raise ValueError("labels must have one entry per column of expr_matrix")
    if size_factors is not None:
        size_factors = np.asarray(size_factors, dtype=float)
        if np.any(size_factors <= 0):
            raise ValueError("All size factors must be > 0.")

    seeds = np.random.SeedSequence(seed).spawn(mat.shape[0])
    jobs = [(mat[i], gene_names[i], labels, size_factors, pairwise_test, pairwise_logfc,
             test_mean, test_dispersion, seeds[i], verbose, kwargs) for i in range(mat.shape[0])]

    qlog = logging.getLogger("qrscore")
    old_level = qlog.level
    if not verbose:
        qlog.setLevel(logging.WARNING)
    try:
        if num_cores > 1:
            with ProcessPoolExecutor(max_workers=num_cores) as ex:
                results = list(ex.map(_one_gene, jobs, chunksize=max(1, len(jobs) // (num_cores * 8))))
        else:
            with warnings.catch_warnings():
                if not verbose:
                    warnings.simplefilter("ignore", RuntimeWarning)
                results = [_one_gene(j) for j in jobs]
    finally:
        qlog.setLevel(old_level)

    return {
        "mean_test": _format([r[0] for r in results], gene_names, "QRscore_Mean",
                             "QRscore_Mean_p_value", rank),
        "var_test": _format([r[1] for r in results], gene_names, "QRscore_Var",
                            "QRscore_Var_p_value", rank),
    }
