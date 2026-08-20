import numpy as np
import pytest

from qgemm.bounds import elementwise_relative_error, gamma_n, measure_u_eff, u_eff_samples
from qgemm.distributions import sample_gaussian
from qgemm.gemm import GemmConfig

# The block-32/E8M0/no-global-scale preset, i.e. MXFP4, with a block size small
# enough that the hand-checked cases below have blocks the eye can follow.
BLOCK4 = GemmConfig(block_size=4, scale_format="e8m0", use_global_scale=False)


def test_gamma_n_matches_definition():
    n, u = 10, 1e-7
    assert gamma_n(n, u) == pytest.approx(n * u / (1 - n * u))


def test_gamma_n_monotonic_in_n():
    u = 1e-7
    assert gamma_n(5, u) < gamma_n(50, u)


def test_gamma_n_rejects_large_nu():
    with pytest.raises(ValueError):
        gamma_n(n=10**10, u=1.0)


# --------------------------------------------------------------------------
# elementwise_relative_error
# --------------------------------------------------------------------------


def test_relative_error_hand_computed_on_a_constant_block():
    # amax = 4.5 = 0.5625 * 2**3 and E2M1's max is 6 = 0.75 * 2**3, so E8M0's
    # round-the-exponent-up rule gives exponent 3 - 3 + 0 = 0, i.e. s_b = 1.
    # 4.5 sits between the grid's 4 and 6, whose midpoint is 5, so it rounds
    # down to 4 and every element carries |4.5 - 4| / 4.5 = 1/9.
    x = np.full(16, 4.5, dtype=np.float64)
    errors = elementwise_relative_error(x, BLOCK4)
    np.testing.assert_allclose(errors, np.full(16, 1.0 / 9.0))


def test_relative_error_is_invariant_to_a_power_of_two_rescaling():
    # 9.0 = 2 * 4.5 forces s_b = 2 instead of 1. An E8M0 scale is a power of
    # two, so the whole recipe is exact under the rescaling and the relative
    # error must be bit-identical to the s_b = 1 case above.
    coarse = elementwise_relative_error(np.full(16, 4.5, dtype=np.float64), BLOCK4)
    scaled = elementwise_relative_error(np.full(16, 9.0, dtype=np.float64), BLOCK4)
    np.testing.assert_array_equal(coarse, scaled)


def test_relative_error_excludes_elements_that_flush_to_zero():
    # One block: amax = 6 gives s_b = 1, so E2M1's smallest nonzero magnitude
    # is 0.5 and everything below 0.25 rounds to exactly zero. The three tiny
    # elements are therefore dropped and only the exactly-represented 6.0
    # survives.
    x = np.array([6.0, 1e-6, 1e-6, 1e-6], dtype=np.float64)
    errors = elementwise_relative_error(x, BLOCK4)
    np.testing.assert_array_equal(errors, np.array([0.0]))


def test_relative_error_keeps_flushed_elements_when_the_threshold_is_zero():
    # The exclusion is a policy, not an accident: turning it off must bring the
    # flushed elements back, each with a relative error of exactly 1.
    x = np.array([6.0, 1e-6, 1e-6, 1e-6], dtype=np.float64)
    errors = elementwise_relative_error(x, BLOCK4, zero_threshold=0.0)
    np.testing.assert_array_equal(errors, np.array([0.0, 1.0, 1.0, 1.0]))


def test_relative_error_drops_exact_zeros_even_with_a_zero_threshold():
    x = np.array([6.0, 0.0, 3.0, 0.0], dtype=np.float64)
    errors = elementwise_relative_error(x, BLOCK4, zero_threshold=0.0)
    np.testing.assert_array_equal(errors, np.array([0.0, 0.0]))


def test_relative_error_rejects_an_unquantized_config():
    with pytest.raises(ValueError, match="quantize"):
        elementwise_relative_error(
            np.full(16, 1.0, dtype=np.float64), GemmConfig(quantize=False, block_size=4)
        )


# --------------------------------------------------------------------------
# u_eff_samples
# --------------------------------------------------------------------------


def test_u_eff_samples_reports_how_many_elements_were_drawn():
    rng = np.random.default_rng(0)
    errors, n_drawn = u_eff_samples(BLOCK4, sample_gaussian, 4096, rng)
    assert n_drawn == 4096
    assert errors.size <= n_drawn  # the near-zero policy only ever removes


def test_u_eff_samples_draws_one_tensor_per_chunk_of_the_requested_shape():
    seen: list[tuple[int, ...]] = []

    def recording_sampler(shape, rng):
        seen.append(shape)
        return sample_gaussian(shape, rng)

    # 100 elements at 64 per tensor rounds up to two whole tensors.
    _, n_drawn = u_eff_samples(
        BLOCK4, recording_sampler, 100, np.random.default_rng(0), tensor_shape=(8, 8)
    )
    assert seen == [(8, 8), (8, 8)]
    assert n_drawn == 128


def test_u_eff_samples_is_reproducible_for_a_given_generator_seed():
    a, _ = u_eff_samples(BLOCK4, sample_gaussian, 4096, np.random.default_rng(7))
    b, _ = u_eff_samples(BLOCK4, sample_gaussian, 4096, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_u_eff_samples_rejects_a_nonpositive_element_count():
    with pytest.raises(ValueError, match="n_elements"):
        u_eff_samples(BLOCK4, sample_gaussian, 0, np.random.default_rng(0))


# --------------------------------------------------------------------------
# measure_u_eff
# --------------------------------------------------------------------------


def test_measure_u_eff_hand_computed_on_a_constant_distribution():
    # Every element carries exactly 1/9 (see the hand-check above), so every
    # quantile of the error distribution is 1/9.
    def constant_sampler(shape, rng):
        return np.full(shape, 4.5, dtype=np.float64)

    u_eff = measure_u_eff(BLOCK4, constant_sampler, 4096, rng=np.random.default_rng(0))
    assert u_eff[0.5] == pytest.approx(1.0 / 9.0)
    assert u_eff[0.99] == pytest.approx(1.0 / 9.0)


def test_measure_u_eff_returns_exactly_the_requested_quantiles():
    u_eff = measure_u_eff(
        BLOCK4, sample_gaussian, 4096, quantiles=(0.25, 0.5, 0.9), rng=np.random.default_rng(0)
    )
    assert sorted(u_eff) == [0.25, 0.5, 0.9]


def test_measure_u_eff_quantiles_are_nondecreasing():
    u_eff = measure_u_eff(
        BLOCK4,
        sample_gaussian,
        1 << 16,
        quantiles=(0.5, 0.9, 0.99),
        rng=np.random.default_rng(0),
    )
    assert u_eff[0.5] <= u_eff[0.9] <= u_eff[0.99]


def test_measure_u_eff_p99_is_stable_when_the_element_count_doubles():
    # The extreme quantile is the one at risk, so it is the one pinned: at this
    # sample size doubling the draw must not move p99 by more than 5%.
    small = measure_u_eff(
        BLOCK4, sample_gaussian, 1 << 16, quantiles=(0.99,), rng=np.random.default_rng(1)
    )
    large = measure_u_eff(
        BLOCK4, sample_gaussian, 1 << 17, quantiles=(0.99,), rng=np.random.default_rng(2)
    )
    assert large[0.99] == pytest.approx(small[0.99], rel=0.05)


def test_measure_u_eff_rejects_quantiles_outside_the_unit_interval():
    with pytest.raises(ValueError, match="quantile"):
        measure_u_eff(BLOCK4, sample_gaussian, 4096, quantiles=(1.5,), rng=np.random.default_rng(0))


def test_measure_u_eff_reports_a_larger_error_for_the_larger_block():
    # One shared scale stretched over more elements leaves more of them far
    # below their block's amax, so block 32 must not resolve better than block
    # 16 under the same scale format. This is the effect the level-gap
    # hypothesis is about; it is pinned here so the sign cannot silently flip.
    common = dict(scale_format="e8m0", use_global_scale=False)
    small = measure_u_eff(
        GemmConfig(block_size=16, **common),
        sample_gaussian,
        1 << 18,
        quantiles=(0.5,),
        rng=np.random.default_rng(3),
    )
    large = measure_u_eff(
        GemmConfig(block_size=32, **common),
        sample_gaussian,
        1 << 18,
        quantiles=(0.5,),
        rng=np.random.default_rng(3),
    )
    assert large[0.5] > small[0.5]
