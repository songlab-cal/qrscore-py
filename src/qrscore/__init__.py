"""QRscore: quantile rank score tests for shifts in mean and dispersion.

Python implementation of the Bioconductor package QRscore (matches R QRscore 1.5.2).
Zhou et al. (2025), Cell Reports Methods 5(9):101147. R names are available as aliases.
"""

import logging

from ._distributions import rzinbinom
from ._models import fit_nb, fit_zinb
from ._utils import computeweight_disp, computeweight_mean, get_composition_pvalue, rank_x
from .core import (qrscore_flex, qrscore_test, qrscore_zinb, qrscore_zinb_nsamples,
                   wilcox_test_pvalue)
from .datasets import load_example_dataset, load_example_dataset_raw
from .de_analysis import p_adjust_bh, qrscore_genetest
from .normalization import estimate_size_factors

logging.getLogger("qrscore").addHandler(logging.NullHandler())

# R-style aliases
QRscore_Flex = qrscore_flex
QRscore_ZINB = qrscore_zinb
QRscore_ZINB_nSamples = qrscore_zinb_nsamples
QRscoreTest = qrscore_test
QRscoreGenetest = qrscore_genetest
getCompositionPValue = get_composition_pvalue

__version__ = "1.5.2"

__all__ = [
    "qrscore_flex", "qrscore_zinb", "qrscore_zinb_nsamples", "qrscore_test",
    "qrscore_genetest", "rank_x", "computeweight_mean", "computeweight_disp",
    "get_composition_pvalue", "rzinbinom", "fit_nb", "fit_zinb", "p_adjust_bh",
    "wilcox_test_pvalue", "load_example_dataset", "load_example_dataset_raw",
    "estimate_size_factors",
    "QRscore_Flex", "QRscore_ZINB", "QRscore_ZINB_nSamples", "QRscoreTest",
    "QRscoreGenetest", "getCompositionPValue",
]
