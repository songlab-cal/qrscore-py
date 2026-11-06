# qrscore (Python)

Python implementation of the Bioconductor package
[QRscore](https://bioconductor.org/packages/QRscore) (Quantile Rank Score):
non-parametric two-sample and K-sample tests for shifts in mean **and**
dispersion of gene expression, with weights derived from NB / ZINB score
functions.

This version matches QRscore **1.5.2** in R. It depends only on numpy, scipy
and pandas.

## Install

```bash
pip install ./qrscore-py          # or: pip install -e ./qrscore-py
pytest qrscore-py/tests           # includes a regression test against R outputs
```

## Quick start

```python
import numpy as np
import qrscore as q

# Two-sample dispersion test with ZINB-derived weights
x = q.rzinbinom(100, size=2, mu=20, pi=0.1, rng=1)
y = q.rzinbinom(100, size=1, mu=20, pi=0.1, rng=2)
q.qrscore_test(np.r_[x, y], np.repeat(["A", "B"], 100),
               measure="dispersion", approx="asymptotic", rng=0)

# Gene-wise analysis on the bundled GTEx example
d = q.load_example_dataset()
res = q.qrscore_genetest(d["example_data2"], d["labels2"],
                         size_factors=d["size_factors2"].values,
                         pairwise_test=True, pairwise_logfc=True,
                         test_mean=True, test_dispersion=True,
                         approx="asymptotic", num_cores=4)
res["var_test"].head()
```

For a full walkthrough on the bundled GTEx data (filtering, size factors,
two- and three-group tests, DEGs and DDGs), see the
[vignette](vignettes/QRscore.md).

## API mapping

| R | Python | R-style alias |
|---|---|---|
| `QRscoreTest` | `qrscore_test` | `QRscoreTest` |
| `QRscoreGenetest` | `qrscore_genetest` | `QRscoreGenetest` |
| `QRscore_Flex` | `qrscore_flex` | `QRscore_Flex` |
| `QRscore_ZINB` | `qrscore_zinb` | `QRscore_ZINB` |
| `QRscore_ZINB_nSamples` | `qrscore_zinb_nsamples` | `QRscore_ZINB_nSamples` |
| `getCompositionPValue` | `get_composition_pvalue` | `getCompositionPValue` |
| `rank_x`, `computeweight_mean`, `computeweight_disp`, `rzinbinom` | same names | |
| `data(example_dataset)` | `load_example_dataset()` | |
| `data(example_dataset_raw_3000_genes)` | `load_example_dataset_raw()` | |
| `DESeq2::estimateSizeFactors` | `estimate_size_factors()` | |

## Notes

* `qrscore_genetest` with two groups computes p-values only when
  `pairwise_test=True` (the default is `False`), as in R.
* In resampling and exact enumeration, the null distribution often has atoms
  exactly at the observed statistic; whether they count as `<` depends on
  floating-point summation order, which can move p by up to ~1% of its value.
* For small samples the dispersion test's model-based variance can exceed the
  variance of the weights actually used (~15% at 200 vs 200), making it
  slightly conservative.
* `perturb` (K-sample) and `n_mom` are accepted but unused, as in R.

## Citation

Zhou F, Aw AJ, Erdmann-Pham DD, Fischer J, Song YS (2025). Robust and adaptive
non-parametric tests for detecting general distributional shifts in gene
expression. *Cell Reports Methods* 5(9):101147.
[doi:10.1016/j.crmeth.2025.101147](https://doi.org/10.1016/j.crmeth.2025.101147)
