import pytest

from qgemm.bounds import gamma_n


def test_gamma_n_matches_definition():
    n, u = 10, 1e-7
    assert gamma_n(n, u) == pytest.approx(n * u / (1 - n * u))


def test_gamma_n_monotonic_in_n():
    u = 1e-7
    assert gamma_n(5, u) < gamma_n(50, u)


def test_gamma_n_rejects_large_nu():
    with pytest.raises(ValueError):
        gamma_n(n=10**10, u=1.0)
