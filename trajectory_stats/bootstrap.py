from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from permutation import _validated_differences

"""
Nick Grebe i6377605

Implementation of the paired bootstrap confidence intervals for mean differnces

"""

@dataclass(frozen=True)
class BootstrapCIResult:
    mean_difference: float
    median_difference: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int
    n_pairs: int


def paired_bootstrap_ci(
    differences: Sequence[float],
    *,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> BootstrapCIResult:
    values = _validated_differences(differences)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    rng = np.random.default_rng(random_seed)
    sample_indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    sample_means = np.mean(values[sample_indices], axis=1)
    alpha = 1.0 - confidence_level
    return BootstrapCIResult(
        mean_difference=float(np.mean(values)),
        median_difference=float(np.median(values)),
        ci_lower=float(np.quantile(sample_means, alpha / 2.0)),
        ci_upper=float(np.quantile(sample_means, 1.0 - alpha / 2.0)),
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        n_pairs=len(values),
    )
