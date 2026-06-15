"""
End-to-end integration tests for blitzgsea.gsea().

All tests use a synthetic 200-gene signature and a small hand-crafted library
so they run offline and complete quickly.  The fixed seed makes results
deterministic.
"""

import numpy as np
import pandas as pd
import pytest

import blitzgsea

# Fast parameters used across all integration tests
_GSEA_KWARGS = dict(
    seed=42,
    processes=1,
    permutations=200,
    anchors=5,
    min_size=5,
    verbose=False,
    progress=False,
)

EXPECTED_COLUMNS = {"es", "nes", "pval", "sidak", "fdr", "geneset_size", "leading_edge"}


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:
    def test_returns_dataframe(self, medium_signature, synthetic_library):
        result = blitzgsea.gsea(medium_signature, synthetic_library, **_GSEA_KWARGS)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, medium_signature, synthetic_library):
        result = blitzgsea.gsea(medium_signature, synthetic_library, **_GSEA_KWARGS)
        assert set(result.columns) == EXPECTED_COLUMNS

    def test_index_name_is_term(self, medium_signature, synthetic_library):
        result = blitzgsea.gsea(medium_signature, synthetic_library, **_GSEA_KWARGS)
        assert result.index.name == "Term"

    def test_all_library_keys_present_in_result(self, medium_signature, synthetic_library):
        result = blitzgsea.gsea(medium_signature, synthetic_library, **_GSEA_KWARGS)
        for key in synthetic_library:
            assert key in result.index

    def test_row_count_matches_valid_sets(self, medium_signature, synthetic_library):
        # All three synthetic sets have ≥ min_size genes, so all three should appear.
        result = blitzgsea.gsea(medium_signature, synthetic_library, **_GSEA_KWARGS)
        assert len(result) == len(synthetic_library)


# ---------------------------------------------------------------------------
# Column dtypes and value ranges
# ---------------------------------------------------------------------------

class TestColumnValues:
    @pytest.fixture(autouse=True)
    def _result(self, medium_signature, synthetic_library):
        self.result = blitzgsea.gsea(medium_signature, synthetic_library, **_GSEA_KWARGS)

    def test_pvals_in_range(self):
        assert (self.result["pval"] >= 0).all()
        assert (self.result["pval"] <= 1).all()

    def test_fdr_in_range(self):
        assert (self.result["fdr"] >= 0).all()
        assert (self.result["fdr"] <= 1).all()

    def test_sidak_in_range(self):
        assert (self.result["sidak"] >= 0).all()
        assert (self.result["sidak"] <= 1).all()

    def test_es_bounded(self):
        # Allow tiny floating-point overshoot at the boundary
        assert (self.result["es"].abs() <= 1.0 + 1e-10).all()

    def test_geneset_size_positive_int(self):
        assert (self.result["geneset_size"] > 0).all()
        assert self.result["geneset_size"].dtype in (int, np.int64, np.int32)

    def test_leading_edge_is_string(self):
        import pandas as pd
        assert pd.api.types.is_string_dtype(self.result["leading_edge"])


# ---------------------------------------------------------------------------
# Enrichment direction
# ---------------------------------------------------------------------------

class TestEnrichmentDirection:
    @pytest.fixture(autouse=True)
    def _result(self, medium_signature, synthetic_library):
        self.result = blitzgsea.gsea(medium_signature, synthetic_library, **_GSEA_KWARGS)

    def test_top_set_has_positive_es(self):
        assert self.result.loc["top_set", "es"] > 0

    def test_bottom_set_has_negative_es(self):
        assert self.result.loc["bottom_set", "es"] < 0

    def test_nes_sign_matches_es_sign(self):
        for term in self.result.index:
            es = self.result.loc[term, "es"]
            nes = self.result.loc[term, "nes"]
            assert np.sign(es) == np.sign(nes), (
                f"ES and NES have different signs for {term}: es={es}, nes={nes}"
            )


# ---------------------------------------------------------------------------
# Gene set size
# ---------------------------------------------------------------------------

class TestGeneSetSize:
    def test_top_set_size(self, medium_signature, synthetic_library):
        result = blitzgsea.gsea(medium_signature, synthetic_library, **_GSEA_KWARGS)
        assert result.loc["top_set", "geneset_size"] == 20

    def test_bottom_set_size(self, medium_signature, synthetic_library):
        result = blitzgsea.gsea(medium_signature, synthetic_library, **_GSEA_KWARGS)
        assert result.loc["bottom_set", "geneset_size"] == 20

    def test_random_set_size(self, medium_signature, synthetic_library):
        result = blitzgsea.gsea(medium_signature, synthetic_library, **_GSEA_KWARGS)
        expected = len(range(0, 200, 13))  # 16 genes
        assert result.loc["random_set", "geneset_size"] == expected


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

class TestSorting:
    def test_sorted_by_pval_ascending(self, medium_signature, synthetic_library):
        result = blitzgsea.gsea(medium_signature, synthetic_library, **_GSEA_KWARGS)
        pvals = list(result["pval"].abs())
        assert pvals == sorted(pvals)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_gives_identical_results(self, medium_signature, synthetic_library):
        r1 = blitzgsea.gsea(medium_signature, synthetic_library, **_GSEA_KWARGS)
        r2 = blitzgsea.gsea(medium_signature, synthetic_library, **_GSEA_KWARGS)
        pd.testing.assert_frame_equal(r1, r2)

    def test_disabling_cache_still_gives_finite_pvals(self, medium_signature, synthetic_library):
        import blitzgsea as blitz
        blitz.pdf_cache.clear()
        result = blitz.gsea(
            medium_signature, synthetic_library,
            **{**_GSEA_KWARGS, "signature_cache": False}
        )
        assert result["pval"].between(0, 1).all()


# ---------------------------------------------------------------------------
# min_size / max_size filtering
# ---------------------------------------------------------------------------

class TestSizeFiltering:
    def test_min_size_excludes_small_sets(self, medium_signature):
        library = {
            "big": [f"GENE_{i}" for i in range(20)],
            "tiny": [f"GENE_{i}" for i in range(3)],
        }
        result = blitzgsea.gsea(
            medium_signature, library, **{**_GSEA_KWARGS, "min_size": 5}
        )
        assert "big" in result.index
        assert "tiny" not in result.index

    def test_max_size_excludes_large_sets(self, medium_signature):
        library = {
            "small": [f"GENE_{i}" for i in range(10)],
            "huge": [f"GENE_{i}" for i in range(100)],
        }
        result = blitzgsea.gsea(
            medium_signature, library, **{**_GSEA_KWARGS, "max_size": 20}
        )
        assert "small" in result.index
        assert "huge" not in result.index


# ---------------------------------------------------------------------------
# Signature cache
# ---------------------------------------------------------------------------

class TestSignatureCache:
    def test_cache_hit_gives_same_result(self, medium_signature, synthetic_library):
        # First call populates the cache; second call should use it and match.
        import blitzgsea as blitz
        blitz.pdf_cache.clear()

        r1 = blitz.gsea(
            medium_signature, synthetic_library,
            **{**_GSEA_KWARGS, "signature_cache": True}
        )
        r2 = blitz.gsea(
            medium_signature, synthetic_library,
            **{**_GSEA_KWARGS, "signature_cache": True}
        )
        pd.testing.assert_frame_equal(r1, r2)
