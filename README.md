# qrscore (Python)

Python implementation of [QRscore](https://bioconductor.org/packages/QRscore):
non-parametric tests that detect shifts in the **mean** and in the
**dispersion** (variance) of gene expression, for two or more groups.

## Install

```bash
pip install git+https://github.com/songlab-cal/qrscore-py
```

Requires Python 3.9+ with numpy, scipy and pandas (installed automatically).

## Quick start

```python
import qrscore as q

d = q.load_example_dataset()          # GTEx whole-blood example

result = q.qrscore_genetest(
    d["example_data2"],               # raw counts, genes x samples
    d["labels2"],                     # age group of each sample
    size_factors=d["size_factors2"],
    pairwise_test=True, test_mean=True, test_dispersion=True,
    approx="asymptotic",
)

result["mean_test"].head()            # genes ranked by mean-shift p-value
result["var_test"].head()             # genes ranked by dispersion-shift p-value
```

For a step-by-step walkthrough (filtering, normalization, two- and
three-group tests, DEGs and DDGs), see the [vignette](vignettes/QRscore.md).

## Main functions

| Function | What it does |
|---|---|
| `qrscore_genetest` | Test every gene in a count matrix for mean and/or dispersion shifts |
| `qrscore_test` | Test a single gene (two or more groups) |
| `estimate_size_factors` | Size factors for normalization (DESeq2 median-of-ratios) |
| `load_example_dataset`, `load_example_dataset_raw` | GTEx whole-blood example data |
| `qrscore_flex`, `qrscore_zinb`, `qrscore_zinb_nsamples` | Lower-level tests with custom or model-based weights |

R function names (`QRscoreTest`, `QRscoreGenetest`, ...) also work.

With two groups, `qrscore_genetest` reports p-values when
`pairwise_test=True`. Use `num_cores` to run genes in parallel.

## Citation

Zhou F, Aw AJ, Erdmann-Pham DD, Fischer J, Song YS (2025). Robust and adaptive
non-parametric tests for detecting general distributional shifts in gene
expression. *Cell Reports Methods* 5(9):101147.
[doi:10.1016/j.crmeth.2025.101147](https://doi.org/10.1016/j.crmeth.2025.101147)
