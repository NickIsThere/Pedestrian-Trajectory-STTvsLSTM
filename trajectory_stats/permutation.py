from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


"""
Nick Grebe i6377605

The implementation of the paired permutation test on model differences
(P value for RQ1)

We should still set up a CI for better reporting

"""

Alternative = Literal["two-sided", "greater", "less"]


@dataclass(frozen=True)
class PermutationTestResult:
    observed_mean_difference: float
    observed_median_difference: float
    p_value: float
    n_pairs: int
    n_permutations: int
    alternative: Alternative


def paired_permutation_test(
    differences: Sequence[float],
    *,
    n_permutations: int = 10000,
    random_seed: int = 42,
    alternative: Alternative = "greater",
) -> PermutationTestResult:
    values = _validated_differences(differences)
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be one of: two-sided, greater, less")

    observed = float(np.mean(values))
    rng = np.random.default_rng(random_seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, len(values)))
    permuted_means = np.mean(signs * values, axis=1)
    if alternative == "greater":
        count_extreme = int(np.sum(permuted_means >= observed))
    elif alternative == "less":
        count_extreme = int(np.sum(permuted_means <= observed))
    else:
        count_extreme = int(np.sum(np.abs(permuted_means) >= abs(observed)))
    return PermutationTestResult(
        observed_mean_difference=observed,
        observed_median_difference=float(np.median(values)),
        p_value=(count_extreme + 1) / (n_permutations + 1),
        n_pairs=len(values),
        n_permutations=n_permutations,
        alternative=alternative,
    )


def _validated_differences(differences: Sequence[float]) -> np.ndarray:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1:
        raise ValueError("differences must be a one-dimensional sequence")
    if len(values) == 0:
        raise ValueError("at least one paired difference is required")
    if not np.isfinite(values).all():
        raise ValueError("paired differences must be finite")
    return values
