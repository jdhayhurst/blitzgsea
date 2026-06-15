"""
Regression tests verifying that each transparent substitution made during
optimisation produces numerically identical results to the original implementation.

Four areas covered:
  1. _score_gene_set  vs  enrichment_score  (O(K) sparse formula vs dense cumsum)
  2. gammainc(a, x/b)  vs  gamma.cdf(x, a, scale=b)
  3. ndtri(1-p)  vs  norm.isf(p)
  4. ThreadPoolExecutor  vs  serial  (processes=1 vs processes=4)
"""

import numpy as np
import polars as pl
import pytest
from scipy.special import gammainc, ndtri
from scipy.stats import gamma, norm

import blitzgsea as blitz
from blitzgsea import _score_gene_set, enrichment_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def medium_abs_sig(rng):
    """200-element abs_signature matching the integration test signature."""
    return np.linspace(5.0, 0.01, 200)


@pytest.fixture
def medium_sig_map():
    return {f"GENE_{i}": i for i in range(200)}


@pytest.fixture
def medium_gene_names():
    return [f"GENE_{i}" for i in range(200)]


@pytest.fixture
def medium_signature_df():
    genes = [f"GENE_{i}" for i in range(200)]
    values = list(np.linspace(5.0, -5.0, 200))
    return pl.DataFrame({"gene": genes, "value": values})


@pytest.fixture
def synthetic_library():
    return {
        "top_set": [f"GENE_{i}" for i in range(20)],
        "bottom_set": [f"GENE_{i}" for i in range(180, 200)],
        "random_set": [f"GENE_{i}" for i in range(0, 200, 13)],
    }


# ---------------------------------------------------------------------------
# 1. _score_gene_set vs enrichment_score
# ---------------------------------------------------------------------------

class TestScoreGeneSetTransparency:
    """_score_gene_set must return the same ES as enrichment_score for any gene set."""

    GENE_SETS = [
        ("top_set",    [f"GENE_{i}" for i in range(20)]),
        ("bottom_set", [f"GENE_{i}" for i in range(180, 200)]),
        ("random_set", [f"GENE_{i}" for i in range(0, 200, 13)]),
        ("singleton",  ["GENE_0"]),
        ("large_set",  [f"GENE_{i}" for i in range(100)]),
    ]

    @pytest.mark.parametrize("name,gene_set", GENE_SETS)
    def test_es_identical(self, name, gene_set, medium_abs_sig, medium_sig_map, medium_gene_names):
        _, es_dense = enrichment_score(medium_abs_sig, medium_sig_map, gene_set)
        es_sparse, _ = _score_gene_set(medium_abs_sig, medium_sig_map, gene_set, medium_gene_names)
        assert es_sparse == pytest.approx(es_dense, rel=1e-6), (
            f"{name}: sparse ES {es_sparse} != dense ES {es_dense}"
        )

    @pytest.mark.parametrize("name,gene_set", GENE_SETS)
    def test_es_sign_identical(self, name, gene_set, medium_abs_sig, medium_sig_map, medium_gene_names):
        _, es_dense = enrichment_score(medium_abs_sig, medium_sig_map, gene_set)
        es_sparse, _ = _score_gene_set(medium_abs_sig, medium_sig_map, gene_set, medium_gene_names)
        if es_dense != 0.0:
            assert np.sign(es_sparse) == np.sign(es_dense)

    def test_leading_edge_genes_are_subset_of_hits(self, medium_abs_sig, medium_sig_map, medium_gene_names):
        gene_set = [f"GENE_{i}" for i in range(20)]
        es, leading_edge = _score_gene_set(medium_abs_sig, medium_sig_map, gene_set, medium_gene_names)
        if leading_edge:
            le_genes = set(leading_edge.split(","))
            assert le_genes.issubset(set(gene_set))

    def test_empty_gene_set_returns_zero(self, medium_abs_sig, medium_sig_map, medium_gene_names):
        es, le = _score_gene_set(medium_abs_sig, medium_sig_map, [], medium_gene_names)
        assert es == 0.0
        assert le == ""

    def test_no_overlap_returns_zero(self, medium_abs_sig, medium_sig_map, medium_gene_names):
        es, le = _score_gene_set(medium_abs_sig, medium_sig_map, ["NOTHERE"], medium_gene_names)
        assert es == 0.0


# ---------------------------------------------------------------------------
# 2. gammainc(a, x/b) vs gamma.cdf(x, a, scale=b)
# ---------------------------------------------------------------------------

class TestGammaIncTransparency:
    """gammainc(a, x/b) must equal gamma.cdf(x, a, scale=b) pointwise."""

    # (x, shape, scale) triples covering small/large shape, tails, middle
    PARAMS = [
        (1.0,  1.5,  1.0),
        (0.5,  0.8,  2.0),
        (10.0, 5.0,  0.5),
        (0.01, 2.0,  1.0),
        (100.0, 20.0, 3.0),
        (1e-6,  1.0,  1.0),
        (50.0,  3.0,  10.0),
    ]

    @pytest.mark.parametrize("x,a,b", PARAMS)
    def test_equals_scipy_gamma_cdf(self, x, a, b):
        expected = gamma.cdf(x, a, scale=b)
        actual = float(gammainc(a, x / b))
        assert actual == pytest.approx(expected, rel=1e-10, abs=1e-15), (
            f"gammainc({a}, {x}/{b}) = {actual}, gamma.cdf = {expected}"
        )

    def test_monotone_in_x(self):
        xs = np.linspace(0.1, 20.0, 50)
        vals = [float(gammainc(2.0, x / 1.5)) for x in xs]
        assert all(v1 <= v2 for v1, v2 in zip(vals, vals[1:]))

    def test_approaches_one_for_large_x(self):
        assert float(gammainc(2.0, 1000.0)) == pytest.approx(1.0, abs=1e-10)

    def test_zero_at_zero(self):
        assert float(gammainc(2.0, 0.0)) == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# 3. ndtri(1-p) vs norm.isf(p)
# ---------------------------------------------------------------------------

class TestNdtriTransparency:
    """ndtri(1-p) must equal norm.isf(p) for all p in (0, 1)."""

    PS = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]

    @pytest.mark.parametrize("p", PS)
    def test_equals_norm_isf(self, p):
        expected = norm.isf(p)
        actual = ndtri(1.0 - p)
        assert actual == pytest.approx(expected, rel=1e-12), (
            f"ndtri(1-{p}) = {actual}, norm.isf({p}) = {expected}"
        )

    def test_symmetry(self):
        for p in [0.05, 0.1, 0.25]:
            assert ndtri(1.0 - p) == pytest.approx(-ndtri(p), rel=1e-12)

    def test_median_is_zero(self):
        assert ndtri(0.5) == pytest.approx(0.0, abs=1e-12)

    def test_monotone_decreasing_in_p(self):
        ps = np.linspace(0.01, 0.99, 100)
        vals = [ndtri(1.0 - p) for p in ps]
        assert all(v1 >= v2 for v1, v2 in zip(vals, vals[1:]))


# ---------------------------------------------------------------------------
# 4. Thread pool transparency: processes=1 vs processes=4
# ---------------------------------------------------------------------------

class TestThreadPoolTransparency:
    """gsea() with processes=4 (ThreadPoolExecutor) must give bit-identical
    results to processes=1 (serial) for the same seed."""

    def _run(self, signature, library, processes, seed=42):
        return blitz.gsea(
            signature,
            library,
            permutations=200,
            seed=seed,
            processes=processes,
            anchors=10,
            ks_disable=True,
        )

    def test_es_identical(self, medium_signature_df, synthetic_library):
        r1 = self._run(medium_signature_df, synthetic_library, processes=1)
        r4 = self._run(medium_signature_df, synthetic_library, processes=4)
        r1 = r1.sort("Term")
        r4 = r4.sort("Term")
        for term in r1["Term"].to_list():
            es1 = r1.filter(pl.col("Term") == term)["es"][0]
            es4 = r4.filter(pl.col("Term") == term)["es"][0]
            assert es1 == pytest.approx(es4, rel=1e-10), f"{term}: es mismatch"

    def test_pval_identical(self, medium_signature_df, synthetic_library):
        r1 = self._run(medium_signature_df, synthetic_library, processes=1)
        r4 = self._run(medium_signature_df, synthetic_library, processes=4)
        r1 = r1.sort("Term")
        r4 = r4.sort("Term")
        for term in r1["Term"].to_list():
            p1 = r1.filter(pl.col("Term") == term)["pval"][0]
            p4 = r4.filter(pl.col("Term") == term)["pval"][0]
            assert p1 == pytest.approx(p4, rel=1e-10), f"{term}: pval mismatch"

    def test_nes_identical(self, medium_signature_df, synthetic_library):
        r1 = self._run(medium_signature_df, synthetic_library, processes=1)
        r4 = self._run(medium_signature_df, synthetic_library, processes=4)
        r1 = r1.sort("Term")
        r4 = r4.sort("Term")
        for term in r1["Term"].to_list():
            n1 = r1.filter(pl.col("Term") == term)["nes"][0]
            n4 = r4.filter(pl.col("Term") == term)["nes"][0]
            assert n1 == pytest.approx(n4, rel=1e-10), f"{term}: nes mismatch"

    def test_leading_edge_identical(self, medium_signature_df, synthetic_library):
        r1 = self._run(medium_signature_df, synthetic_library, processes=1)
        r4 = self._run(medium_signature_df, synthetic_library, processes=4)
        r1 = r1.sort("Term")
        r4 = r4.sort("Term")
        assert r1["leading_edge"].to_list() == r4["leading_edge"].to_list()

    def test_different_seeds_still_match_across_processes(self, medium_signature_df, synthetic_library):
        for seed in [0, 7, 99]:
            r1 = self._run(medium_signature_df, synthetic_library, processes=1, seed=seed)
            r4 = self._run(medium_signature_df, synthetic_library, processes=4, seed=seed)
            r1 = r1.sort("Term")
            r4 = r4.sort("Term")
            assert r1["es"].to_list() == r4["es"].to_list(), f"seed={seed}: es mismatch"
