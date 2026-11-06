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

Argument renames: `wList` → `w_list`, `LR.test` → `lr_test`, `gene.name` →
`gene_name`, `ties.break` → `ties_break`, `pairwise_logFC` → `pairwise_logfc`,
`bigN` → `big_n`. Every function that draws random numbers takes
`rng` (a `numpy.random.Generator` or seed). `NA` is returned as `nan`;
R `message()`s go to the `"qrscore"` logger; R warnings are Python warnings.

## How closely it matches R

Model fitting replicates the R code paths step by step rather than calling a
generic optimiser, because results are sensitive to *where* a fit stops:

* NB fit — `MASS::glm.nb` (Poisson start, `theta.ml` Newton iterations,
  alternation loop), then QRscore's check against a direct maximisation
  (R's `optim` BFGS with its finite-difference gradient, ported exactly).
* `pscl::zeroinfl` (negbin) — same truncated-count pre-fit, starting values and
  BFGS optimiser (`vmmin`) with pscl's tolerance.
* `dnbinom` — R's saddle-point algorithm (`dbinom_raw`/`bd0`/`stirlerr`);
  `qnbinom` with R's left-continuity fuzz; `wilcox.test(correct = FALSE)`.

Validated against QRscore 1.5.2 in R (R 4.3.3, MASS 7.3-60, pscl 1.5.9) on
2,880 reference values (scripts in `validation/`):

* Weight vectors and variances: agree to ~1e-11.
* Asymptotic tests, Mann–Whitney, NB-weighted two- and K-sample tests: agree
  to ~1e-7.
* ZINB-weighted tests: within 1e-3, usually exact (on very flat likelihoods the
  BFGS path can differ in the last floating-point bits).
* `QRscoreGenetest` on 150 GTEx genes (2 and 3 groups, mean and dispersion):
  log fold changes to 1e-14, p-values within 0.002.
* Resampling p-values agree within Monte Carlo error; random streams differ from R.

For exact comparisons `rank_x`'s random tie-breaking is replaced by a
deterministic rule on both sides; with random tie-breaking, p-values for tied
count data vary between runs in R as well.

## Options beyond the R package

* `fit_method="mle"` (default `"r"`): fit the NB / ZINB models by a
  multi-start maximum-likelihood search instead of reproducing `MASS::glm.nb`
  and `pscl::zeroinfl` step by step.

## Other differences from the R package

* **Parallelism:** `num_cores` uses processes; each gene gets its own random
  stream from `seed`, so results do not depend on `num_cores`.
* **Label order:** sorted with Python ordering (R's `sort()` uses locale
  collation, which can differ for mixed-case labels).

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
