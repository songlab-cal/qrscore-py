# Quantile Rank Score (QRscore): Quick Start in Python

## Introduction

Differential expression (DE) analysis, aiming to identify genes with
significant expression changes across conditions, provides insights into
molecular mechanisms underlying aging, disease, and other biological processes.
Traditional DE methods primarily detect changes in centrality (e.g., mean or
median), but often lack power against alternative hypotheses characterized by
changes in spread (e.g. variance or dispersion). Variance shifts, however, are
critical in understanding regulatory dynamics and stochasticity in gene
expression, particularly in contexts like aging and cellular differentiation.
Moreover, in DE analysis, there is often a trade-off between statistical power
and control over the false discovery rate (FDR): parametric approaches may
inflate FDRs, while nonparametric methods frequently lack sufficient power.

`qrscore` addresses these two limitations by providing a robust framework for
two-sample and K-sample tests that detect shifts in both centrality and spread.
It extends the Mann-Whitney test to an adaptive, rank-based approach that
combines non-parametric tests with weights informed by (zero-inflated) negative
binomial models, ensuring both high power and strictly controlled FDR.

This is the Python implementation of the Bioconductor package
[QRscore](https://bioconductor.org/packages/QRscore); it gives the same results
as QRscore 1.5.2 in R. This vignette follows the R vignette step by step.

## Installation

```bash
pip install git+https://github.com/songlab-cal/qrscore-py
```

## Overview

This vignette illustrates how to:

1. Preprocess and normalize bulk RNA-seq data.
2. Perform two-sample and three-sample differential tests using `qrscore`.
3. Obtain results for DEGs and DDGs.

## Analysis

### Package Loading

```python
import numpy as np
import pandas as pd
import qrscore as q
```

### Load the data

The example dataset contains 3000 randomly selected genes in RNA-seq counts
data from whole blood samples in the GTEx project, along with associated
metadata for age groups.

```python
raw = q.load_example_dataset_raw()
bulk_mat = raw["COUNTS"]                      # samples x genes, raw counts
print(bulk_mat.shape)
```

Age groups are aggregated for larger sample size and simplified analysis.

```python
ages = raw["METADATA"]["AGE"].replace({
    "20-29": "20-39", "30-39": "20-39",
    "40-49": "40-59", "50-59": "40-59",
    "60-69": "60-79", "70-79": "60-79",
}).to_numpy()
pd.Series(ages).value_counts().sort_index().rename("samples").to_frame()
```

### Data Prefiltering and Normalization

* Genes with low expression or high dropout rates are excluded.
* Size factors are estimated with DESeq2's median-of-ratios method
  (`q.estimate_size_factors` gives the same values as
  `DESeq2::estimateSizeFactors`).

```python
col_means = bulk_mat.mean()
col_zeros = (bulk_mat == 0).mean()
# The following thresholds can be modified
kept = (col_means > 5) & (col_zeros < 0.2)

expr_mat = bulk_mat.loc[:, kept].T            # genes x samples
est_size_factors = q.estimate_size_factors(expr_mat)
print("Number of kept genes after prefiltering:", expr_mat.shape[0])
```

### Two-sample test

#### Define Age Groups for Testing

This example compares the 20-39 and 60-79 age groups.

```python
kept_samples = np.isin(ages, ["20-39", "60-79"])
expr_mat_1 = expr_mat.loc[:, kept_samples]    # directly use raw counts
est_size_factors_1 = est_size_factors[kept_samples].to_numpy()
ages_1 = ages[kept_samples]
```

#### Run QRscore Test

Both mean and dispersion shifts are tested in parallel. The outputs include
ranked p-values of differential variance and differential mean tests for
all genes, together with log ratios of mean shifts and variance shifts.

```python
results_2_sample = q.qrscore_genetest(
    expr_mat_1, ages_1,
    size_factors=est_size_factors_1,
    pairwise_test=True,
    pairwise_logfc=True, test_mean=True,
    test_dispersion=True, num_cores=2,
    approx="asymptotic",
)
```

On macOS and Windows, run code that uses `num_cores > 1` from a notebook or
under `if __name__ == "__main__":` in a script.

#### Differentially Expressed Genes (DEGs)

Genes with significant mean shifts are identified.

```python
results_2_sample_mean = results_2_sample["mean_test"]
results_2_sample_DEG = results_2_sample_mean[
    results_2_sample_mean["QRscore_Mean_adj_p_value"] < 0.05
]
print("Number of DEGs:", len(results_2_sample_DEG))
results_2_sample_DEG.head()
```

#### Differentially Dispersed Genes (DDGs)

Genes with significant variance shifts are identified.

```python
results_2_sample_var = results_2_sample["var_test"]
results_2_sample_DDG = results_2_sample_var[
    results_2_sample_var["QRscore_Var_adj_p_value"] < 0.05
]
print("Number of DDGs:", len(results_2_sample_DDG))
results_2_sample_DDG.head()
```

### Three-Sample Test

This example compares the 20-39, 40-59, and 60-79 age groups.

The output includes 3-sample test p-values as well as pairwise fold changes.
To get a more comprehensive result, one can set `pairwise_test=True` to
additionally obtain the pairwise test p-values.

```python
results_3_sample = q.qrscore_genetest(
    expr_mat, ages, size_factors=est_size_factors.to_numpy(),
    pairwise_test=False,
    pairwise_logfc=True, test_mean=True,
    test_dispersion=True, num_cores=2,
    approx="asymptotic",
)
```

For detecting DDGs and DEGs, it's recommended to use 3-sample test p-values
(namely `QRscore_Mean_adj_p_value` and `QRscore_Var_adj_p_value`) with certain
cutoffs (e.g. 0.05).

```python
### DEGs
results_3_sample_mean = results_3_sample["mean_test"]
results_3_sample_DEG = results_3_sample_mean[
    results_3_sample_mean["QRscore_Mean_adj_p_value"] < 0.05
]
print("Number of DEGs:", len(results_3_sample_DEG))
results_3_sample_DEG.head()
```

```python
### DDGs
results_3_sample_var = results_3_sample["var_test"]
results_3_sample_DDG = results_3_sample_var[
    results_3_sample_var["QRscore_Var_adj_p_value"] < 0.05
]
print("Number of DDGs:", len(results_3_sample_DDG))
results_3_sample_DDG.head()
```

## Session Info

```python
import platform, scipy
print("Python", platform.python_version(), "|", platform.platform())
print("qrscore", q.__version__, "| numpy", np.__version__,
      "| scipy", scipy.__version__, "| pandas", pd.__version__)
```
