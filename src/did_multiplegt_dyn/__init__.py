"""
did_multiplegt_dyn package initialization.
All common imports are made available here for convenience.
"""

# ============================================================
# === POLARS / NUMPY / CORE PYTHON ===
# ============================================================
import polars as pl
import numpy as np
import math
import warnings

# ============================================================
# === TYPING UTILITIES ===
# ============================================================
from typing import (
    Any,
    List,
    Union,
    Optional,
    Sequence,
    Dict,
    Iterable,
)

# ============================================================
# === SCIPY ===
# ============================================================
from scipy.stats import norm, chi2, t as student_t

# ============================================================
# === STATSMODELS ===
# ============================================================
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.stats.diagnostic import linear_harvey_collier
from statsmodels.stats.contrast import ContrastResults
from statsmodels.stats.sandwich_covariance import cov_hc0, cov_hc1

# ============================================================
# === PANDAS ===
# ============================================================
import pandas as pd

# ============================================================
# === INTERNAL PACKAGE IMPORTS ===
# ============================================================
from .did_multiplegt_dyn_core import did_multiplegt_dyn_core_pl
from ._utils import *
from .did_multiplegt_dyn import DidMultiplegtDyn

# ============================================================
# === PUBLIC EXPORTS (USER-FACING API) ===
# ============================================================
__all__ = [
    # core libs
    "pl", "np", "math", "warnings",
    
    # typing
    "Any", "List", "Union", "Optional", "Sequence", "Dict", "Iterable",

    # scipy
    "norm", "chi2", "student_t",

    # statsmodels
    "sm", "smf",
    "summary_table",
    "linear_harvey_collier",
    "ContrastResults",
    "cov_hc0", "cov_hc1",

    # pandas
    "pd",

    # internal exports
    "did_multiplegt_dyn_core_pl",
    "DidMultiplegtDyn",
]