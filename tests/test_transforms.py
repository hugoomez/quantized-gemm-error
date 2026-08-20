import numpy as np
import pytest
from scipy.stats import kurtosis

from qgemm.transforms import (
    RHT_BLOCK_SIZES,
    apply_rht,
    hadamard_matrix,
    invert_rht,
    random_signs,
)


def _randomized_hadamard(block_size: int, seed: int) -> np.ndarray:
    """The matrix `apply_rht` applies, built from its public parts: H' = H @ diag(eps)."""
    signs = random_signs(block_size, np.random.default_rng(seed))
    return hadamard_matrix(block_size) * signs


# --- the Hadamard matrix itself -----------------------------------------------


@pytest.mark.parametrize("block_size", RHT_BLOCK_SIZES)
def test_hadamard_matrix_is_orthogonal(block_size):
    h = hadamard_matrix(block_size)

    assert h.shape == (block_size, block_size)
    assert h.dtype == np.float64
    assert np.allclose(h @ h.T, np.eye(block_size), atol=1e-12)


@pytest.mark.parametrize("block_size", RHT_BLOCK_SIZES)
def test_randomizing_the_signs_preserves_orthogonality(block_size):
    h = _randomized_hadamard(block_size, seed=0)

    assert np.allclose(h @ h.T, np.eye(block_size), atol=1e-12)


def test_hadamard_matrix_follows_the_sylvester_recursion():
    # H_1 = [1]; H_2n = (1/sqrt(2)) * [[H_n, H_n], [H_n, -H_n]].
    assert hadamard_matrix(1) == np.array([[1.0]])

    h8 = hadamard_matrix(8)
    h4 = hadamard_matrix(4)
    top = np.hstack([h4, h4])
    bottom = np.hstack([h4, -h4])
    assert np.allclose(h8, np.vstack([top, bottom]) / np.sqrt(2), atol=1e-15)


def test_hadamard_entries_all_have_the_same_magnitude():
    # Every entry is +-1/sqrt(block_size): the transform spreads a spike evenly.
    for block_size in RHT_BLOCK_SIZES:
        h = hadamard_matrix(block_size)
        assert np.allclose(np.abs(h), 1.0 / np.sqrt(block_size), atol=1e-15)


def test_random_signs_are_plus_or_minus_one_and_come_from_the_passed_generator():
    signs = random_signs(32, np.random.default_rng(7))

    assert signs.shape == (32,)
    assert signs.dtype == np.float64
    assert set(np.unique(signs)) <= {-1.0, 1.0}
    # Same seed -> same draw; the caller controls the state, not global NumPy.
    assert np.array_equal(signs, random_signs(32, np.random.default_rng(7)))
    assert not np.array_equal(signs, random_signs(32, np.random.default_rng(8)))


# --- the inner product is preserved -------------------------------------------


@pytest.mark.parametrize("block_size", RHT_BLOCK_SIZES)
def test_transformed_vectors_have_the_same_inner_product(block_size):
    rng = np.random.default_rng(123)
    a = rng.standard_normal(block_size)
    b = rng.standard_normal(block_size)

    # Both operands must see the SAME sign draw, so both generators start equal.
    ta = apply_rht(a, block_size, np.random.default_rng(0))
    tb = apply_rht(b, block_size, np.random.default_rng(0))

    assert ta @ tb == pytest.approx(a @ b, rel=1e-14, abs=1e-14)


def test_the_invariant_needs_both_operands_to_share_the_sign_draw():
    # The misuse this guards against: transforming the two GEMM operands with
    # generators in different states silently breaks (H'a).(H'b) == a.b.
    rng = np.random.default_rng(5)
    a = rng.standard_normal(32)
    b = rng.standard_normal(32)

    ta = apply_rht(a, 32, np.random.default_rng(0))
    tb = apply_rht(b, 32, np.random.default_rng(1))

    assert ta @ tb != pytest.approx(a @ b, rel=1e-6)


def test_inner_product_is_preserved_blockwise_across_a_whole_matmul():
    rng = np.random.default_rng(11)
    a = rng.standard_normal((6, 64))
    b = rng.standard_normal((5, 64))

    ta = apply_rht(a, 16, np.random.default_rng(3))
    tb = apply_rht(b, 16, np.random.default_rng(3))

    assert np.allclose(ta @ tb.T, a @ b.T, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("block_size", RHT_BLOCK_SIZES)
def test_the_transform_is_norm_preserving(block_size):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 4 * block_size))

    out = apply_rht(x, block_size, np.random.default_rng(9))

    assert np.allclose(np.linalg.norm(out, axis=-1), np.linalg.norm(x, axis=-1), atol=1e-12)


# --- round trip ---------------------------------------------------------------


@pytest.mark.parametrize("block_size", RHT_BLOCK_SIZES)
def test_invert_rht_undoes_apply_rht(block_size):
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3, 2, 3 * block_size))

    out = apply_rht(x, block_size, np.random.default_rng(17))
    back = invert_rht(out, block_size, np.random.default_rng(17))

    assert np.allclose(back, x, rtol=1e-13, atol=1e-13)


def test_invert_rht_with_a_different_generator_does_not_round_trip():
    rng = np.random.default_rng(6)
    x = rng.standard_normal(32)

    out = apply_rht(x, 32, np.random.default_rng(0))
    back = invert_rht(out, 32, np.random.default_rng(1))

    assert not np.allclose(back, x, atol=1e-6)


# --- outlier dispersion -------------------------------------------------------


def test_a_lone_outlier_is_spread_evenly_over_its_block():
    # [100, 0, ..., 0] of length 16: every output component is exactly
    # +-100/sqrt(16) = +-25, so the block amax drops from 100 to 25.
    x = np.zeros(16, dtype=np.float64)
    x[0] = 100.0

    out = apply_rht(x, 16, np.random.default_rng(0))

    assert np.abs(x).max() == 100.0
    assert np.array_equal(np.abs(out), np.full(16, 25.0))
    assert np.abs(out).max() == 25.0


def test_the_outlier_stays_spread_wherever_it_sits_in_the_block():
    for position in range(16):
        x = np.zeros(16, dtype=np.float64)
        x[position] = 100.0

        out = apply_rht(x, 16, np.random.default_rng(position))

        assert np.array_equal(np.abs(out), np.full(16, 25.0))


def test_each_block_is_transformed_independently():
    # A spike in block 0 must not leak into block 1.
    x = np.zeros(32, dtype=np.float64)
    x[0] = 100.0

    out = apply_rht(x, 16, np.random.default_rng(0))

    assert np.array_equal(np.abs(out[:16]), np.full(16, 25.0))
    assert np.array_equal(out[16:], np.zeros(16))


# --- kurtosis -----------------------------------------------------------------


def test_rht_reduces_the_kurtosis_of_a_heavy_tailed_sample():
    # Direction only: t-Student with nu=2 has infinite population kurtosis, so
    # no exact target exists. The RHT mixes each block toward a Gaussian, and
    # the point of the test is that the empirical kurtosis falls a lot.
    rng = np.random.default_rng(20260820)
    x = rng.standard_t(df=2, size=(256, 32))

    out = apply_rht(x, 32, np.random.default_rng(1))

    before = kurtosis(x, axis=None, fisher=False)
    after = kurtosis(out, axis=None, fisher=False)
    assert after < before / 2


def test_rht_lowers_the_typical_per_block_amax_of_a_heavy_tailed_sample():
    # The practical consequence for block quantization: a smaller block amax
    # means a smaller shared scale and finer resolution for the whole block.
    rng = np.random.default_rng(20260820)
    x = rng.standard_t(df=2, size=(512, 32))

    out = apply_rht(x, 32, np.random.default_rng(1))

    assert np.median(np.abs(out).max(axis=-1)) < np.median(np.abs(x).max(axis=-1))


# --- input contract -----------------------------------------------------------


@pytest.mark.parametrize("block_size", [0, 3, 6, 12, 24, 48, 100, -8])
def test_non_power_of_two_block_sizes_are_rejected(block_size):
    x = np.zeros(96, dtype=np.float64)

    with pytest.raises(ValueError, match="power of two"):
        apply_rht(x, block_size, np.random.default_rng(0))
    with pytest.raises(ValueError, match="power of two"):
        invert_rht(x, block_size, np.random.default_rng(0))
    with pytest.raises(ValueError, match="power of two"):
        hadamard_matrix(block_size)


def test_a_last_axis_that_is_not_a_whole_number_of_blocks_is_rejected():
    x = np.zeros((2, 40), dtype=np.float64)

    with pytest.raises(ValueError, match="multiple of block_size"):
        apply_rht(x, 16, np.random.default_rng(0))


def test_non_float64_input_is_rejected():
    x = np.zeros(32, dtype=np.float32)

    with pytest.raises(TypeError, match="float64"):
        apply_rht(x, 16, np.random.default_rng(0))
    with pytest.raises(TypeError, match="float64"):
        invert_rht(x, 16, np.random.default_rng(0))


def test_output_is_float64_and_keeps_the_input_shape():
    x = np.random.default_rng(0).standard_normal((3, 5, 64))

    out = apply_rht(x, 32, np.random.default_rng(0))

    assert out.dtype == np.float64
    assert out.shape == x.shape


def test_the_project_block_sizes_are_the_documented_four():
    assert RHT_BLOCK_SIZES == (8, 16, 32, 64)


# --- short-row fallback (block_size > n) ---------------------------------------
# Previously crashed outright (ValueError: "not a multiple of block_size") for
# every (n, block_size) pair with n < block_size, including this project's own
# (n=16, block_size=32, rht=True) grid cells -- untested by Step 1.6's original
# suite, which never exercised block_size > n.


@pytest.mark.parametrize("n,block_size", [(16, 32), (8, 16), (4, 64), (1, 8)])
def test_a_row_shorter_than_block_size_does_not_crash(n, block_size):
    x = np.random.default_rng(0).standard_normal(n)

    out = apply_rht(x, block_size, np.random.default_rng(0))

    assert out.dtype == np.float64
    assert out.shape == x.shape


@pytest.mark.parametrize("n,block_size", [(16, 32), (8, 16), (4, 64)])
def test_short_row_fallback_preserves_the_inner_product(n, block_size):
    rng = np.random.default_rng(123)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)

    # Both operands must see the same sign draw, as in the normal case.
    ta = apply_rht(a, block_size, np.random.default_rng(0))
    tb = apply_rht(b, block_size, np.random.default_rng(0))

    assert ta @ tb == pytest.approx(a @ b, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("n,block_size", [(16, 32), (8, 16), (4, 64)])
def test_short_row_fallback_is_orthogonal(n, block_size):
    # H' @ H'.T == I at the fallback size, the same property asserted for the
    # normal case by test_randomizing_the_signs_preserves_orthogonality.
    h = _randomized_hadamard(n, seed=0)

    assert np.allclose(h @ h.T, np.eye(n), atol=1e-12)


@pytest.mark.parametrize("n,block_size", [(16, 32), (8, 16), (4, 64)])
def test_short_row_fallback_round_trips(n, block_size):
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3, n))

    out = apply_rht(x, block_size, np.random.default_rng(17))
    back = invert_rht(out, block_size, np.random.default_rng(17))

    assert np.allclose(back, x, rtol=1e-12, atol=1e-12)


def test_short_row_fallback_uses_the_row_length_not_the_nominal_block_size():
    # A lone spike at n=16 with block_size=32 must spread over the 16
    # elements actually present -- 100/sqrt(16) = 25 -- not over a
    # nonexistent 32-element block (100/sqrt(32) ~= 17.68). This is the
    # documented reduction in spread power, not just a shape accommodation.
    x = np.zeros(16, dtype=np.float64)
    x[3] = 100.0

    out = apply_rht(x, 32, np.random.default_rng(0))

    assert np.allclose(np.abs(out), 100.0 / np.sqrt(16), atol=1e-12)
    assert not np.allclose(np.abs(out), 100.0 / np.sqrt(32), atol=1e-12)


def test_short_row_fallback_matches_calling_with_block_size_equal_to_n():
    # apply_rht(x, 32, rng) on a length-16 row must be identical to calling
    # it with block_size=16 directly: the fallback is exactly "use n".
    seed = 9
    x = np.random.default_rng(1).standard_normal((5, 16))

    via_fallback = apply_rht(x, 32, np.random.default_rng(seed))
    via_explicit = apply_rht(x, 16, np.random.default_rng(seed))

    assert np.array_equal(via_fallback, via_explicit)


def test_short_row_fallback_requires_the_row_length_itself_be_a_power_of_two():
    # block_size=32 > n=20 falls back to n=20, which is not a power of two --
    # the fallback size gets the same validation as any other block size,
    # not a free pass just because it came from a shape rather than an
    # argument.
    x = np.zeros(20, dtype=np.float64)

    with pytest.raises(ValueError, match="power of two"):
        apply_rht(x, 32, np.random.default_rng(0))


def test_a_row_longer_than_block_size_and_not_a_multiple_still_raises():
    # Regression guard: the short-row fallback must not swallow the genuine
    # trailing-partial-block case (n > block_size, not a whole multiple),
    # which has no unambiguous fallback and stays deliberately unchanged
    # (see apply_rht's "Tail policy").
    x = np.zeros(100, dtype=np.float64)

    with pytest.raises(ValueError, match="multiple of block_size"):
        apply_rht(x, 32, np.random.default_rng(0))
