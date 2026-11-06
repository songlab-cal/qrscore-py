"""Size factors for count normalisation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_size_factors(counts):
    """DESeq2 median-of-ratios size factors (as ``DESeq2::estimateSizeFactors``).

    Parameters
    ----------
    counts : pandas.DataFrame or 2-D array
        Raw counts, genes x samples.

    Returns
    -------
    pandas.Series (indexed by sample) or numpy array: one size factor per sample.
    """
    mat = counts.to_numpy(dtype=float) if isinstance(counts, pd.DataFrame) else np.asarray(counts, float)
    with np.errstate(divide="ignore"):
        logc = np.log(mat)
    log_geo_means = logc.mean(axis=1)                 # -inf for genes with any zero
    use = np.isfinite(log_geo_means)
    if not np.any(use):
        raise ValueError("every gene contains a zero in at least one sample; "
                         "cannot compute median-of-ratios size factors")
    sf = np.empty(mat.shape[1])
    for j in range(mat.shape[1]):
        ok = use & (mat[:, j] > 0)
        sf[j] = np.exp(np.median(logc[ok, j] - log_geo_means[ok]))
    if isinstance(counts, pd.DataFrame):
        return pd.Series(sf, index=counts.columns, name="size_factor")
    return sf
