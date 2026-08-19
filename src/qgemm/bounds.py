"""Classical (Higham-style) floating-point error bounds.

Reference: N. J. Higham, "Accuracy and Stability of Numerical Algorithms",
2nd ed., SIAM, 2002.
"""

from __future__ import annotations


def gamma_n(n: int, u: float) -> float:
    """Higham's gamma_n(u) = n*u / (1 - n*u).

    Standard bound on the relative error growth from `n` sequential
    floating-point roundoffs at unit roundoff `u`. This is a conservative
    (worst-case, sequential-summation) bound: algorithms with better
    dependency structure (e.g. pairwise/cascade summation, as used by
    `numpy.sum`) satisfy strictly smaller error bounds, so an empirical
    error exceeding `gamma_n` indicates a real problem, not just a loose
    bound.
    """
    if n * u >= 1:
        raise ValueError("n * u must be < 1 for gamma_n to be defined")
    return n * u / (1 - n * u)
