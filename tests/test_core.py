"""Tests for the core algorithm functions in blitzgsea/__init__.py."""

import numpy as np
import polars as pl
import pytest

import blitzgsea


# ---------------------------------------------------------------------------
# strip_gene_set
# ---------------------------------------------------------------------------

class TestStripGeneSet:
    def test_filters_absent_genes(self):
        sig_genes = {"A", "B", "C"}
        result = blitzgsea.strip_gene_set(sig_genes, ["A", "B", "X", "Y"])
        assert result == ["A", "B"]

    def test_all_present(self):
        sig_genes = {"A", "B", "C"}
        result = blitzgsea.strip_gene_set(sig_genes, ["A", "B", "C"])
        assert set(result) == {"A", "B", "C"}

    def test_none_present(self):
        sig_genes = {"A", "B"}
        result = blitzgsea.strip_gene_set(sig_genes, ["X", "Y", "Z"])
        assert result == []

    def test_empty_gene_set(self):
        sig_genes = {"A", "B"}
        assert blitzgsea.strip_gene_set(sig_genes, []) == []

    def test_empty_signature_genes(self):
        assert blitzgsea.strip_gene_set(set(), ["A", "B"]) == []


# ---------------------------------------------------------------------------
# clean_library
# ---------------------------------------------------------------------------

class TestCleanLibrary:
    def test_filters_to_signature(self, sig_df_10):
        library = {
            "set_a": {f"GENE_{i}" for i in range(5)},
            "set_b": {"GENE_0", "UNKNOWN_1", "UNKNOWN_2"},
        }
        result = blitzgsea.clean_library(library, sig_df_10)
        assert result["set_a"] == {f"GENE_{i}" for i in range(5)}
        assert result["set_b"] == {"GENE_0"}

    def test_preserves_all_valid_genes(self, sig_df_10):
        all_genes = {f"GENE_{i}" for i in range(10)}
        library = {"full_set": all_genes}
        result = blitzgsea.clean_library(library, sig_df_10)
        assert result["full_set"] == all_genes

    def test_returns_empty_set_for_no_overlap(self, sig_df_10):
        library = {"no_match": {"UNKNOWN_A", "UNKNOWN_B"}}
        result = blitzgsea.clean_library(library, sig_df_10)
        assert result["no_match"] == set()

    def test_preserves_all_keys(self, sig_df_10):
        library = {"a": {"GENE_0"}, "b": {"GENE_1"}, "c": {"UNKNOWN"}}
        result = blitzgsea.clean_library(library, sig_df_10)
        assert set(result.keys()) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# enrichment_score
# ---------------------------------------------------------------------------

class TestEnrichmentScore:
    def test_top_genes_positive_es(self, abs_sig_10, sig_map_10):
        # Top 3 genes → all hit increments pile up first → ES = 1.0
        _, es = blitzgsea.enrichment_score(
            abs_sig_10, sig_map_10, {"GENE_0", "GENE_1", "GENE_2"}
        )
        assert es == pytest.approx(1.0)

    def test_bottom_genes_negative_es(self, abs_sig_10, sig_map_10):
        # Bottom 3 genes → running sum dips first → ES ≈ -1.0
        _, es = blitzgsea.enrichment_score(
            abs_sig_10, sig_map_10, {"GENE_7", "GENE_8", "GENE_9"}
        )
        assert es == pytest.approx(-1.0, abs=1e-10)

    def test_returns_tuple(self, abs_sig_10, sig_map_10):
        result = blitzgsea.enrichment_score(
            abs_sig_10, sig_map_10, {"GENE_0", "GENE_1"}
        )
        assert isinstance(result, tuple) and len(result) == 2

    def test_running_sum_length(self, abs_sig_10, sig_map_10):
        rs, _ = blitzgsea.enrichment_score(
            abs_sig_10, sig_map_10, {"GENE_0", "GENE_1"}
        )
        assert len(rs) == len(abs_sig_10)

    def test_running_sum_ends_near_zero(self, abs_sig_10, sig_map_10):
        # By construction the cumsum normalises so the final value ≈ 0.
        rs, _ = blitzgsea.enrichment_score(
            abs_sig_10, sig_map_10, {"GENE_0", "GENE_1", "GENE_2"}
        )
        assert rs[-1] == pytest.approx(0.0, abs=1e-10)

    def test_es_bounded_by_one(self, abs_sig_10, sig_map_10):
        for gene_set in [
            {"GENE_0"},
            {"GENE_5"},
            {"GENE_9"},
            {"GENE_0", "GENE_4", "GENE_9"},
        ]:
            _, es = blitzgsea.enrichment_score(abs_sig_10, sig_map_10, gene_set)
            # Allow tiny floating-point overshoot at the boundary (e.g. -1.0 - 2e-16)
            assert abs(es) <= 1.0 + 1e-10

    def test_genes_missing_from_map_are_ignored(self, abs_sig_10, sig_map_10):
        _, es = blitzgsea.enrichment_score(
            abs_sig_10, sig_map_10, {"GENE_0", "NOT_IN_SIG"}
        )
        # Should still run; NOT_IN_SIG is silently dropped
        assert -1.0 <= es <= 1.0


# ---------------------------------------------------------------------------
# enrichment_score_null
# ---------------------------------------------------------------------------

class TestEnrichmentScoreNull:
    def test_seeded_is_reproducible(self, abs_sig_10):
        hit_indicator = np.zeros(10)
        np.random.seed(42)
        es1 = blitzgsea.enrichment_score_null(abs_sig_10, hit_indicator.copy(), 3)
        np.random.seed(42)
        es2 = blitzgsea.enrichment_score_null(abs_sig_10, hit_indicator.copy(), 3)
        assert es1 == es2

    def test_returns_scalar(self, abs_sig_10):
        hit_indicator = np.zeros(10)
        np.random.seed(0)
        es = blitzgsea.enrichment_score_null(abs_sig_10, hit_indicator.copy(), 3)
        assert np.isscalar(es) or es.ndim == 0

    def test_es_in_valid_range(self, abs_sig_10):
        hit_indicator = np.zeros(10)
        for seed in range(10):
            np.random.seed(seed)
            es = blitzgsea.enrichment_score_null(abs_sig_10, hit_indicator.copy(), 3)
            assert -1.0 <= float(es) <= 1.0

    def test_different_seeds_give_different_results(self, abs_sig_10):
        hit_indicator = np.zeros(10)
        results = set()
        for seed in range(20):
            np.random.seed(seed)
            es = blitzgsea.enrichment_score_null(abs_sig_10, hit_indicator.copy(), 3)
            results.add(round(float(es), 6))
        # With 20 different seeds we expect more than one unique value
        assert len(results) > 1


# ---------------------------------------------------------------------------
# get_leading_edge
# ---------------------------------------------------------------------------

class TestGetLeadingEdge:
    def test_positive_es_returns_genes_before_peak(self, abs_sig_10, sig_map_10, gene_names_10):
        # Top-3 gene set → peak at index 2, leading edge = genes at indices 0 and 1
        rs, _ = blitzgsea.enrichment_score(
            abs_sig_10, sig_map_10, {"GENE_0", "GENE_1", "GENE_2"}
        )
        le = blitzgsea.get_leading_edge(
            rs, gene_names_10, ["GENE_0", "GENE_1", "GENE_2"], sig_map_10
        )
        assert set(le.split(",")) == {"GENE_0", "GENE_1"}

    def test_negative_es_returns_genes_after_trough(self, abs_sig_10, sig_map_10, gene_names_10):
        # Bottom-3 gene set → trough at index 6, leading edge = genes at indices 7,8,9
        rs, _ = blitzgsea.enrichment_score(
            abs_sig_10, sig_map_10, {"GENE_7", "GENE_8", "GENE_9"}
        )
        le = blitzgsea.get_leading_edge(
            rs, gene_names_10, ["GENE_7", "GENE_8", "GENE_9"], sig_map_10
        )
        assert set(le.split(",")) == {"GENE_7", "GENE_8", "GENE_9"}

    def test_returns_string(self, abs_sig_10, sig_map_10, gene_names_10):
        rs, _ = blitzgsea.enrichment_score(
            abs_sig_10, sig_map_10, {"GENE_0", "GENE_1"}
        )
        le = blitzgsea.get_leading_edge(
            rs, gene_names_10, ["GENE_0", "GENE_1"], sig_map_10
        )
        assert isinstance(le, str)


# ---------------------------------------------------------------------------
# loess_interpolation
# ---------------------------------------------------------------------------

class TestLoessInterpolation:
    def test_returns_callable(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        f = blitzgsea.loess_interpolation(x, y)
        assert callable(f)

    def test_interpolates_linear_data(self):
        x = np.linspace(1, 10, 20)
        y = 3.0 * x + 1.0
        f = blitzgsea.loess_interpolation(x, y)
        for xi in [2.0, 5.0, 8.0]:
            assert f(xi) == pytest.approx(3.0 * xi + 1.0, rel=0.05)

    def test_extrapolates_without_error(self):
        x = np.linspace(1, 10, 20)
        y = 2.0 * x
        f = blitzgsea.loess_interpolation(x, y)
        val = f(20.0)
        assert np.isfinite(val)

    def test_vectorised_input(self):
        x = np.linspace(1, 10, 20)
        y = x ** 2
        f = blitzgsea.loess_interpolation(x, y)
        result = f(np.array([3.0, 5.0, 7.0]))
        assert result.shape == (3,)
