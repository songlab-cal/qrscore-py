"""Bundled GTEx whole-blood example data (R: data(example_dataset))."""

from __future__ import annotations

from importlib import resources

import pandas as pd


def load_example_dataset():
    """Return the example dataset as a dict.

    Keys mirror the R object: ``example_data1`` / ``example_data2`` (gene x sample
    count DataFrames), ``labels1`` / ``labels2`` (age groups) and
    ``size_factors1`` / ``size_factors2`` (pandas Series indexed by sample).
    Set 1 has three age groups (20-39, 40-59, 60-79); set 2 has 20-39 and 60-79.
    """
    root = resources.files("qrscore") / "data"
    out = {}
    for i in (1, 2):
        with resources.as_file(root / f"example_data{i}.csv.gz") as f:
            out[f"example_data{i}"] = pd.read_csv(f, index_col=0)
        with resources.as_file(root / f"example_samples{i}.csv.gz") as f:
            s = pd.read_csv(f)
        out[f"labels{i}"] = s["label"].to_numpy()
        out[f"size_factors{i}"] = pd.Series(s["size_factor"].to_numpy(), index=s["sample"])
    return out
