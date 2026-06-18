"""
Regression tests verifying that each transparent substitution made during
optimisation produces numerically identical results to the original implementation.

Six areas covered:
  1. _score_gene_set  vs  enrichment_score  (O(K) sparse formula vs dense cumsum)
  2. gammainc(a, x/b)  vs  gamma.cdf(x, a, scale=b)
  3. ndtri(1-p)  vs  norm.isf(p)
  4. ThreadPoolExecutor  vs  serial  (processes=1 vs processes=4)
  5. get_peak_size_adv O(K) null distribution vs O(N) reference (np.random.choice)
  6. gsea() ES and leading_edge vs direct enrichment_score() calls (end-to-end)
"""

import numpy as np
import polars as pl
import pytest
from scipy.special import gammainc, ndtri
from scipy.stats import gamma, ks_2samp, norm

import blitzgsea as blitz
from blitzgsea import (
    _score_gene_set,
    enrichment_score,
    get_leading_edge,
    get_peak_size_adv,
)


def _make_hs(sig_map: dict, gene_set: list) -> np.ndarray:
    """Convert a gene-name list to a sorted int64 index array for _score_gene_set."""
    return np.array(sorted(sig_map[x] for x in gene_set if x in sig_map), dtype=np.int64)


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
        es_sparse, _ = _score_gene_set(medium_abs_sig, _make_hs(medium_sig_map, gene_set), medium_gene_names)
        assert es_sparse == pytest.approx(es_dense, rel=1e-6), (
            f"{name}: sparse ES {es_sparse} != dense ES {es_dense}"
        )

    @pytest.mark.parametrize("name,gene_set", GENE_SETS)
    def test_es_sign_identical(self, name, gene_set, medium_abs_sig, medium_sig_map, medium_gene_names):
        _, es_dense = enrichment_score(medium_abs_sig, medium_sig_map, gene_set)
        es_sparse, _ = _score_gene_set(medium_abs_sig, _make_hs(medium_sig_map, gene_set), medium_gene_names)
        if es_dense != 0.0:
            assert np.sign(es_sparse) == np.sign(es_dense)

    def test_leading_edge_genes_are_subset_of_hits(self, medium_abs_sig, medium_sig_map, medium_gene_names):
        gene_set = [f"GENE_{i}" for i in range(20)]
        es, leading_edge = _score_gene_set(medium_abs_sig, _make_hs(medium_sig_map, gene_set), medium_gene_names)
        if leading_edge:
            le_genes = set(leading_edge.split(","))
            assert le_genes.issubset(set(gene_set))

    def test_empty_gene_set_returns_zero(self, medium_abs_sig, medium_sig_map, medium_gene_names):
        es, le = _score_gene_set(medium_abs_sig, _make_hs(medium_sig_map, []), medium_gene_names)
        assert es == 0.0
        assert le == ""

    def test_no_overlap_returns_zero(self, medium_abs_sig, medium_sig_map, medium_gene_names):
        es, le = _score_gene_set(medium_abs_sig, _make_hs(medium_sig_map, ["NOTHERE"]), medium_gene_names)
        assert es == 0.0

    def test_leading_edge_zero_abs_sig_tie(self):
        """When a hit gene has abs_sig=0, the dense argmax ties at-hit vs before-hit
        and picks the smaller index (before-hit). The sparse formula must agree."""
        gene_names = [f"G{i}" for i in range(6)]
        sig_map = {f"G{i}": i for i in range(6)}
        abs_sig = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
        gene_set = ["G2", "G5"]  # G5 has abs_sig=0 → triggers tie at positions 4 and 5

        rs, es_dense = enrichment_score(abs_sig, sig_map, gene_set)
        le_dense = get_leading_edge(rs, gene_names, gene_set, sig_map)
        es_sparse, le_sparse = _score_gene_set(abs_sig, _make_hs(sig_map, gene_set), gene_names)

        assert es_sparse == pytest.approx(es_dense, rel=1e-6), (
            f"zero-abs-sig tie: sparse ES {es_sparse} != dense ES {es_dense}"
        )
        assert np.sign(es_sparse) == np.sign(es_dense)
        dense_set = set(le_dense.split(",")) if le_dense else set()
        sparse_set = set(le_sparse.split(",")) if le_sparse else set()
        assert dense_set == sparse_set, (
            f"zero-abs-sig tie: dense leading edge {dense_set} != sparse {sparse_set}"
        )

    def test_leading_edge_matches_dense_random(self):
        """Compare _score_gene_set leading edge against enrichment_score+get_leading_edge
        over 200 random (abs_sig, gene_set) pairs covering varied shapes and sizes."""
        rng = np.random.default_rng(7)
        N = 200
        gene_names = [f"G{i}" for i in range(N)]
        sig_map = {f"G{i}": i for i in range(N)}
        failures = []

        for trial in range(200):
            abs_sig = np.abs(rng.normal(0, 1, N))
            K = int(rng.integers(2, 60))
            gene_set = [f"G{i}" for i in rng.choice(N, K, replace=False).tolist()]

            rs, _ = enrichment_score(abs_sig, sig_map, gene_set)
            le_dense = get_leading_edge(rs, gene_names, gene_set, sig_map)
            _, le_sparse = _score_gene_set(abs_sig, _make_hs(sig_map, gene_set), gene_names)

            dense_set = set(le_dense.split(",")) if le_dense else set()
            sparse_set = set(le_sparse.split(",")) if le_sparse else set()

            if dense_set != sparse_set:
                failures.append((trial, K, dense_set, sparse_set))

        assert not failures, (
            f"{len(failures)}/200 trials had mismatched leading edges.\n"
            + "\n".join(
                f"  trial={t} K={k}: dense={d} sparse={s}"
                for t, k, d, s in failures[:5]
            )
        )


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
            max_workers=processes,
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


# ---------------------------------------------------------------------------
# 5. Null distribution: O(K) get_peak_size_adv vs O(N) reference
# ---------------------------------------------------------------------------

def _reference_null_es(
    abs_signature: np.ndarray, K: int, permutations: int, seed: int
) -> list[float]:
    """O(N) reference null-ES using np.random.choice + dense cumsum.

    Mirrors the original get_peak_size_adv implementation that was in the
    codebase before the O(K) exponential-spacings rewrite.  Used as the
    ground-truth oracle in statistical equivalence tests.
    """
    rng = np.random.default_rng(seed)
    N = len(abs_signature)
    number_miss = N - K
    norm_no_hit = 1.0 / number_miss
    es_out: list[float] = []
    for _ in range(permutations):
        hits = rng.choice(N, size=K, replace=False)
        total = float(abs_signature[hits].sum())
        if total == 0.0:
            continue
        norm_hit = 1.0 / total
        h_ind = np.zeros(N, dtype=np.float64)
        h_ind[hits] = 1.0
        rs = np.cumsum(
            h_ind * abs_signature * norm_hit - (1.0 - h_ind) * norm_no_hit
        )
        es_out.append(float(rs[np.abs(rs).argmax()]))
    return es_out


class TestNullDistributionTransparency:
    """get_peak_size_adv (O(K) exponential-spacings) must produce ES samples
    from the same distribution as the O(N) reference (np.random.choice +
    dense cumsum).  Any divergence here means calibration parameters
    (alpha, beta) will be wrong, corrupting every p-value."""

    @pytest.mark.parametrize("K,N", [(10, 200), (30, 500), (80, 1000)])
    def test_distribution_matches_reference_ks(self, K, N):
        """Two-sample KS test: O(K) and O(N) null ES samples must be drawn
        from the same distribution (p > 0.01 with 3000 permutations each)."""
        rng = np.random.default_rng(77)
        abs_sig = np.sort(np.abs(rng.standard_normal(N)))[::-1]

        permutations = 3000
        es_fast = get_peak_size_adv(abs_sig, K, permutations, seed=0)
        es_ref = _reference_null_es(abs_sig, K, permutations, seed=0)

        stat, pval = ks_2samp(es_fast, es_ref)
        assert pval > 0.01, (
            f"K={K}, N={N}: KS test failed (stat={stat:.4f}, p={pval:.4f}). "
            f"O(K) null distribution differs from O(N) reference."
        )

    def test_null_mean_near_zero_constant_weights(self):
        """With uniform abs values every hit has equal weight, so the null
        distribution is symmetric and its mean must be near 0."""
        abs_sig = np.ones(400)  # constant weights → symmetric null
        es = np.array(get_peak_size_adv(abs_sig, 40, 3000, seed=42))
        assert abs(es.mean()) < 0.05, f"Null mean {es.mean():.4f} far from 0"

    def test_pos_neg_balance_constant_weights(self):
        """Uniform abs values → ~50% positive and ~50% negative null ES."""
        abs_sig = np.ones(300)
        es = np.array(get_peak_size_adv(abs_sig, 30, 3000, seed=3))
        pos_frac = (es > 0).mean()
        assert 0.35 < pos_frac < 0.65, (
            f"Expected ~50% positive null ES with uniform weights, got {pos_frac:.2%}"
        )

    def test_moments_match_reference(self):
        """Mean and std of O(K) null must be within 5% of O(N) reference."""
        N, K = 400, 35
        rng = np.random.default_rng(55)
        abs_sig = np.sort(np.abs(rng.standard_normal(N)))[::-1]

        permutations = 4000
        es_fast = np.array(get_peak_size_adv(abs_sig, K, permutations, seed=9))
        es_ref = np.array(_reference_null_es(abs_sig, K, permutations, seed=9))

        assert abs(es_fast.mean() - es_ref.mean()) < 0.05 * (abs(es_ref.mean()) + 1e-6), (
            f"Mean mismatch: fast={es_fast.mean():.4f}, ref={es_ref.mean():.4f}"
        )
        assert abs(es_fast.std() - es_ref.std()) < 0.05 * es_ref.std() + 1e-6, (
            f"Std mismatch: fast={es_fast.std():.4f}, ref={es_ref.std():.4f}"
        )


# ---------------------------------------------------------------------------
# 6. End-to-end: gsea() ES and leading_edge vs direct enrichment_score()
# ---------------------------------------------------------------------------

def _gsea_preprocessed_signature(signature_df: pl.DataFrame):
    """Replicate gsea()'s internal signature preprocessing (sort, dedupe, center)."""
    sig = signature_df.rename(
        {signature_df.columns[0]: "i", signature_df.columns[1]: "v"}
    )
    sig = sig.sort("v", descending=True).unique(
        subset=["i"], keep="first", maintain_order=True
    )
    sig = sig.with_columns((pl.col("v") - pl.col("v").mean()).alias("v"))
    gene_names = sig["i"].to_list()
    abs_sig = sig["v"].abs().to_numpy()
    sig_map = {g: i for i, g in enumerate(gene_names)}
    return gene_names, abs_sig, sig_map


class TestEndToEndESTransparency:
    """gsea() ES and leading_edge must agree with direct enrichment_score() /
    get_leading_edge() calls on the same preprocessed signature.

    Catches bugs in preprocessing (centering, deduplication) or in
    _score_gene_set that only manifest inside the full pipeline."""

    def _run_gsea(self, signature_df, library):
        return blitz.gsea(
            signature_df,
            library,
            permutations=200,
            seed=42,
            anchors=5,
            ks_disable=True,
        )

    def test_es_matches_enrichment_score(self, medium_signature_df, synthetic_library):
        """Every term's ES in gsea() output equals enrichment_score() directly."""
        result = self._run_gsea(medium_signature_df, synthetic_library)
        gene_names, abs_sig, sig_map = _gsea_preprocessed_signature(medium_signature_df)
        sig_genes = set(gene_names)

        for row in result.iter_rows(named=True):
            term = row["Term"]
            stripped = blitz.strip_gene_set(sig_genes, synthetic_library[term])
            _, es_direct = enrichment_score(abs_sig, sig_map, stripped)
            assert row["es"] == pytest.approx(es_direct, rel=1e-6), (
                f"{term}: gsea es={row['es']:.6f} != enrichment_score={es_direct:.6f}"
            )

    def test_leading_edge_matches_get_leading_edge(self, medium_signature_df, synthetic_library):
        """Every term's leading_edge set in gsea() equals get_leading_edge()."""
        result = self._run_gsea(medium_signature_df, synthetic_library)
        gene_names, abs_sig, sig_map = _gsea_preprocessed_signature(medium_signature_df)
        sig_genes = set(gene_names)

        for row in result.iter_rows(named=True):
            term = row["Term"]
            stripped = blitz.strip_gene_set(sig_genes, synthetic_library[term])
            rs, _ = enrichment_score(abs_sig, sig_map, stripped)
            le_direct = get_leading_edge(rs, gene_names, stripped, sig_map)
            le_direct_set = set(le_direct.split(",")) if le_direct else set()
            le_gsea_set = set(row["leading_edge"].split(",")) if row["leading_edge"] else set()
            assert le_gsea_set == le_direct_set, (
                f"{term}: gsea leading_edge={le_gsea_set} != get_leading_edge={le_direct_set}"
            )

    def test_es_stable_across_seeds(self, medium_signature_df, synthetic_library):
        """ES values must be identical regardless of seed (ES is deterministic,
        only calibration/p-values depend on the null distribution seed)."""
        r0 = self._run_gsea(medium_signature_df, synthetic_library).sort("Term")
        r1 = blitz.gsea(
            medium_signature_df,
            synthetic_library,
            permutations=200,
            seed=99,
            anchors=5,
            ks_disable=True,
        ).sort("Term")
        assert r0["es"].to_list() == r1["es"].to_list(), (
            "ES changed between seed=42 and seed=99 — ES must be deterministic"
        )
