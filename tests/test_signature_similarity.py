"""Tests for blitzgsea/signature_similarity.py."""

import numpy as np
import pytest

from blitzgsea.signature_similarity import (
    best_kl_fit,
    create_pdf,
    kl_divergence,
    map_density_range,
)


class TestCreatePDF:
    def test_output_shapes_match(self):
        data = np.random.default_rng(0).normal(size=500)
        xv, pdf = create_pdf(data, bins=100)
        assert xv.shape == (100,)
        assert pdf.shape == (100,)

    def test_pdf_non_negative(self):
        data = np.random.default_rng(1).normal(size=500)
        _, pdf = create_pdf(data, bins=100)
        assert np.all(pdf >= 0)

    def test_x_values_span_data_range(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        xv, _ = create_pdf(data, bins=50)
        assert xv[0] == pytest.approx(1.0)
        assert xv[-1] == pytest.approx(5.0)

    def test_default_bins(self):
        data = np.random.default_rng(2).normal(size=200)
        xv, pdf = create_pdf(data)
        assert len(xv) == 200
        assert len(pdf) == 200


class TestKLDivergence:
    def test_identical_distributions_zero(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_different_distributions_positive(self):
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        assert kl_divergence(p, q) > 0

    def test_zero_entries_ignored(self):
        # Entries where p=0 or q=0 should not contribute (or crash).
        p = np.array([1.0, 0.0])
        q = np.array([0.5, 0.5])
        val = kl_divergence(p, q)
        assert np.isfinite(val)

    def test_asymmetric(self):
        # Use genuinely asymmetric distributions (not mirror-symmetric, which gives equal KL).
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.1, 0.5, 0.4])
        assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p))


class TestMapDensityRange:
    def test_output_length_matches_new_range(self):
        old_range = np.linspace(0, 1, 100)
        old_pdf = np.ones(100) / 100
        new_range = np.linspace(0.2, 0.8, 50)
        result = map_density_range(new_range, old_range, old_pdf)
        assert result.shape == (50,)

    def test_out_of_range_fills_zero(self):
        old_range = np.linspace(0, 1, 100)
        old_pdf = np.ones(100)
        new_range = np.array([-1.0, 2.0])
        result = map_density_range(new_range, old_range, old_pdf)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)

    def test_identity_mapping(self):
        old_range = np.linspace(0, 1, 50)
        old_pdf = old_range ** 2
        result = map_density_range(old_range, old_range, old_pdf)
        np.testing.assert_allclose(result, old_pdf, rtol=1e-10)


class TestBestKLFit:
    def test_returns_zero_divergence_for_self(self):
        data = np.random.default_rng(0).normal(size=500)
        xv, pdf = create_pdf(data, bins=100)
        pdfs = {"sig_a": {"xvalues": xv, "pdf": pdf}}
        kld, key = best_kl_fit(data, pdfs, bins=100)
        assert kld == pytest.approx(0.0, abs=1e-3)
        assert key == "sig_a"

    def test_selects_closest_match(self):
        # Use overlapping distributions so the KL mask retains valid entries.
        # data_a ~ N(0,1), data_b ~ N(3,1) — different means but overlapping tails.
        rng = np.random.default_rng(42)
        data_a = rng.normal(loc=0, size=1000)
        data_b = rng.normal(loc=3, size=1000)

        xv_a, pdf_a = create_pdf(data_a, bins=200)
        xv_b, pdf_b = create_pdf(data_b, bins=200)
        pdfs = {
            "near": {"xvalues": xv_a, "pdf": pdf_a},
            "far": {"xvalues": xv_b, "pdf": pdf_b},
        }

        # Query drawn from the same distribution as data_a → should match "near"
        query = rng.normal(loc=0, size=1000)
        _, key = best_kl_fit(query, pdfs, bins=200)
        assert key == "near"

    def test_returns_tuple(self):
        data = np.random.default_rng(1).normal(size=200)
        xv, pdf = create_pdf(data, bins=50)
        result = best_kl_fit(data, {"k": {"xvalues": xv, "pdf": pdf}}, bins=50)
        assert isinstance(result, tuple) and len(result) == 2
