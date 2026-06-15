"""Tests for blitzgsea/mpsci.py — high-precision math utilities."""

import math
import numpy as np
import pytest
from scipy.stats import gamma as scipy_gamma, norm as scipy_norm

from blitzgsea.mpsci import gammacdf, invcdf


class TestGammaCDF:
    def test_matches_scipy_at_moderate_values(self):
        # For values away from the extremes both implementations should agree closely.
        cases = [
            (1.0, 1.0, 1.0),
            (2.0, 3.0, 1.0),
            (5.0, 2.0, 2.0),
            (0.5, 0.5, 1.0),
        ]
        for x, k, theta in cases:
            expected = scipy_gamma.cdf(x, k, scale=theta)
            actual = float(gammacdf(x, k, theta))
            assert actual == pytest.approx(expected, rel=1e-10), (
                f"gammacdf({x}, {k}, {theta}): got {actual}, expected {expected}"
            )

    def test_returns_probability(self):
        for x in [0.1, 1.0, 5.0, 10.0]:
            p = float(gammacdf(x, 2.0, 1.0))
            assert 0.0 <= p <= 1.0

    def test_negative_input_returns_zero(self):
        assert float(gammacdf(-1.0, 2.0, 1.0)) == 0.0

    def test_monotone_increasing(self):
        xs = [0.5, 1.0, 2.0, 5.0, 10.0]
        probs = [float(gammacdf(x, 2.0, 1.0)) for x in xs]
        assert all(probs[i] < probs[i + 1] for i in range(len(probs) - 1))

    def test_extreme_precision_does_not_raise(self):
        # A very large x should approach 1 without raising or returning nan.
        p = float(gammacdf(100.0, 1.0, 1.0))
        assert math.isfinite(p)
        assert p > 0.999


class TestInvCDF:
    def test_median_is_zero(self):
        assert invcdf(0.5) == pytest.approx(0.0, abs=1e-10)

    def test_symmetry_around_half(self):
        # The function mirrors values around 0.5, so invcdf(p) == -invcdf(1-p).
        for p in [0.1, 0.2, 0.3, 0.4]:
            assert invcdf(p) == pytest.approx(-invcdf(1.0 - p), abs=1e-10)

    def test_known_quantiles(self):
        # invcdf(0.025) should equal the 97.5th normal percentile (~1.96) because
        # the function maps small p-values to large positive NES magnitudes.
        assert invcdf(0.025) == pytest.approx(1.9599639845400545, rel=1e-6)
        assert invcdf(0.975) == pytest.approx(-1.9599639845400545, rel=1e-6)

    def test_output_is_finite(self):
        for p in [0.001, 0.01, 0.5, 0.99, 0.999]:
            assert math.isfinite(invcdf(p))

    def test_near_boundary_is_finite(self):
        # Very small or very large (but not 0/1) p-values should be finite.
        assert math.isfinite(invcdf(0.001))
        assert math.isfinite(invcdf(0.999))
