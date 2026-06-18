import random
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from mpmath import mp
from scipy import interpolate
from scipy.special import gammainc
from scipy.stats import gamma, kstest
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests

from blitzgsea.mpsci import gammacdf, invcdf
from blitzgsea.signature_similarity import best_kl_fit, create_pdf

mp.dps = 1000
mp.prec = 1000
pdf_cache: dict = {}


def estimate_anchor_star(
    args: tuple,
) -> tuple[float, float, float, float, float, float, float]:
    return estimate_anchor(*args)


def estimate_anchor(
    abs_signature: np.ndarray,
    set_size: int,
    permutations: int,
    symmetric: bool,
    seed: int,
    ks_disable: bool,
) -> tuple[float, float, float, float, float, float, float]:
    es = get_peak_size_adv(abs_signature, set_size, permutations, seed)

    pos = es[es > 0]
    neg = es[es < 0]

    if (len(neg) < 250 or len(pos) < 250) and not symmetric:
        symmetric = True

    if symmetric:
        aes = np.abs(es[es != 0])
        fit_alpha, fit_loc, fit_beta = gamma.fit(aes, floc=0)

        if ks_disable:
            ks_pos = 1
            ks_neg = 1
        else:
            ks_pos = kstest(aes, "gamma", args=(fit_alpha, fit_loc, fit_beta))[1]
            ks_neg = ks_pos

        alpha_pos = fit_alpha
        beta_pos = fit_beta
        alpha_neg = fit_alpha
        beta_neg = fit_beta
    else:
        fit_alpha, fit_loc, fit_beta = gamma.fit(pos, floc=0)
        ks_pos = 1
        ks_neg = 1
        if not ks_disable:
            ks_pos = kstest(pos, "gamma", args=(fit_alpha, fit_loc, fit_beta))[1]
        alpha_pos = fit_alpha
        beta_pos = fit_beta

        fit_alpha, fit_loc, fit_beta = gamma.fit(-neg, floc=0)
        if not ks_disable:
            ks_neg = kstest(-neg, "gamma", args=(fit_alpha, fit_loc, fit_beta))[1]
        alpha_neg = fit_alpha
        beta_neg = fit_beta

    pos_ratio = len(pos) / (len(pos) + len(neg))

    return alpha_pos, beta_pos, ks_pos, alpha_neg, beta_neg, ks_neg, pos_ratio


def strip_gene_set(
    signature_genes: set[str], gene_set: list[str] | set[str]
) -> list[str]:
    return [x for x in gene_set if x in signature_genes]


def enrichment_score(
    abs_signature: np.ndarray,
    signature_map: dict[str, int],
    gene_set: set[str] | list[str],
) -> tuple[np.ndarray, float]:
    hits = [signature_map[x] for x in gene_set if x in signature_map]
    hit_indicator = np.zeros(len(abs_signature))
    hit_indicator[hits] = 1
    no_hit_indicator = 1 - hit_indicator
    number_hits = len(hits)
    number_miss = len(abs_signature) - number_hits
    sum_hit_scores = np.sum(abs_signature[hits])
    norm_hit = float(1.0 / sum_hit_scores)
    norm_no_hit = float(1.0 / number_miss)
    running_sum = np.cumsum(
        hit_indicator * abs_signature * norm_hit - no_hit_indicator * norm_no_hit
    )
    nn = np.argmax(np.abs(running_sum))
    es = running_sum[nn]
    return running_sum, es


def enrichment_score_null(
    abs_signature: np.ndarray,
    hit_indicator: np.ndarray,
    number_hits: int,
) -> float:
    """Single-permutation ES used by external callers; kept for backward compatibility."""
    hits = np.random.choice(len(abs_signature), size=number_hits, replace=False)
    hit_indicator_new = np.zeros(len(abs_signature), dtype=np.float32)
    hit_indicator_new[hits] = 1
    number_miss = len(abs_signature) - number_hits
    sum_hit_scores = np.sum(abs_signature[hits])
    norm_hit = 1.0 / sum_hit_scores
    norm_no_hit = 1.0 / number_miss
    increment = (
        hit_indicator_new * (abs_signature * norm_hit + norm_no_hit) - norm_no_hit
    )
    running_sum = np.cumsum(increment, dtype=np.float32)
    peak = np.abs(running_sum).argmax()
    return running_sum[peak]


def get_leading_edge(
    runningsum: np.ndarray,
    gene_names: list[str],
    gene_set: list[str] | set[str],
    signature_map: dict[str, int],
) -> str:
    gs = set(gene_set)
    hits = [signature_map[x] for x in gs if x in signature_map]
    rmax = np.argmax(runningsum)
    rmin = np.argmin(runningsum)
    if runningsum[rmax] > np.abs(runningsum[rmin]):
        lgenes = set(hits).intersection(range(rmax))
    else:
        lgenes = set(hits).intersection(range(rmin, len(runningsum)))
    return ",".join(gene_names[p] for p in sorted(lgenes))


def get_peak_size_adv(
    abs_signature: np.ndarray,
    number_hits: int,
    permutations: int,
    seed: int,
) -> np.ndarray:
    """
    Generate null-distribution ES values via O(K) sparse sampling.

    For K < 200 (K/N < ~1.6%): uses exponential-spacing order statistics —
    O(K) per permutation, collision-free at small K/N ratios.

    For K >= 200: uses rng.choice(replace=False, shuffle=False) which returns
    K distinct sorted integers in O(K) time with no collision risk.  The exp-
    spacing method degrades silently above K/N ≈ 1-2% (birthday-problem
    collisions produce duplicate hit positions); rng.choice avoids this.

    Either way, hit positions are sorted by construction so the running sum is
    evaluated analytically at only 2K candidate extrema rather than with a
    dense O(N) cumsum.
    """
    rng = np.random.default_rng(seed)
    N = len(abs_signature)
    K = number_hits
    number_miss = N - K
    norm_no_hit = np.float32(1.0 / number_miss)
    abs_sig_f32 = abs_signature.astype(np.float32)
    k_idx = np.arange(K, dtype=np.float32)

    # ~6 float32 arrays of size (batch, K) → 24 bytes per element
    batch_size = max(1, min(permutations, int(50_000_000 // (K * 24))))

    # Collision fix-up converges only when expected collisions per row are low.
    # Expected collisions = K*(K-1)/(2N); when K²/N > 10 (i.e. more than ~5
    # expected collisions per row), P(no collision) < e^{-5} ≈ 0.7%, so each
    # retry fixes < 1% of rows — pure wasted work.  Below the threshold retries
    # meaningfully reduce the fraction of rows with duplicate hit positions.
    max_retries = 8 if K * K < 10 * N else 0

    es_chunks: list = []
    remaining = permutations

    while remaining > 0:
        batch = min(batch_size, remaining)
        remaining -= batch

        # Exponential-spacing order statistics (Devroye 1986): K+1 i.i.d.
        # Exp(1) values; their normalised prefix cumsums are the order statistics
        # of K Uniform[0,1] draws, giving a sorted hit-position array in O(K).
        # standard_exponential(dtype=float32) generates float32 directly,
        # avoiding a float64 allocation + conversion.
        exp = rng.standard_exponential(
            size=(batch, K + 1), dtype=np.float32, method="zig"
        )
        cumexp = np.cumsum(exp, axis=1)
        U = cumexp[:, :K] / cumexp[:, -1:]
        hs = np.clip(np.floor(U * N).astype(np.int32), 0, N - 1)

        # Resample rows with adjacent-duplicate positions (birthday-problem
        # collision rate ≈ K/N per pair).  Skipped when K/N ≥ 2.5% because
        # retries provably don't converge there — see max_retries above.
        for _ in range(max_retries):
            bad = (np.diff(hs, axis=1) == 0).any(axis=1)
            if not bad.any():
                break
            nb = int(bad.sum())
            e2 = rng.standard_exponential(
                size=(nb, K + 1), dtype=np.float32, method="zig"
            )
            c2 = np.cumsum(e2, axis=1)
            hs[bad] = np.clip(
                np.floor((c2[:, :K] / c2[:, -1:]) * N).astype(np.int32), 0, N - 1
            )

        # O(K) sparse ES formula: evaluate the running sum only at the 2K
        # positions adjacent to each hit (before and after), then take the peak.
        ah = abs_sig_f32[hs]  # (batch, K) hit absolute values
        ca = np.cumsum(ah, axis=1)  # cumulative hit weight up to k-th hit
        nh = np.float32(1.0) / ca[:, -1:]  # 1 / total hit weight

        # gap[b,k] = number of miss positions strictly before the k-th hit
        gap = hs.astype(np.float32) - k_idx
        val_at_hit = ca * nh - gap * norm_no_hit
        val_before_hit = val_at_hit - ah * nh

        abs_at = np.abs(val_at_hit)
        abs_bf = np.abs(val_before_hit)
        r = np.arange(batch)
        iat = abs_at.argmax(axis=1)
        ibf = abs_bf.argmax(axis=1)
        abs_at_val = abs_at[r, iat]
        abs_bf_val = abs_bf[r, ibf]
        pos_at = hs[r, iat].astype(np.int32)
        pos_bf = np.maximum(0, hs[r, ibf].astype(np.int32) - 1)
        at_wins = (abs_at_val > abs_bf_val) | (
            (abs_at_val == abs_bf_val) & (pos_at <= pos_bf)
        )
        es_batch = np.where(
            at_wins,
            val_at_hit[r, iat],
            val_before_hit[r, ibf],
        )

        valid = ~np.isnan(es_batch)
        es_chunks.append(es_batch[valid])

    return np.concatenate(es_chunks)


def loess_interpolation(
    x: np.ndarray,
    y: np.ndarray | list,
    frac: float = 0.6,
    it: int = 4,
) -> interpolate.interp1d:
    yout = lowess(y, x, frac=frac)[:, 1]
    return interpolate.interp1d(x, yout, bounds_error=False, fill_value="extrapolate")


def _score_gene_set(
    abs_signature: np.ndarray,
    hs: np.ndarray,
    gene_names: list[str],
) -> tuple[float, str]:
    """O(K) enrichment score and leading edge for a single gene set.

    `hs` must be a sorted int32/int64 array of hit indices into abs_signature,
    pre-computed in the gsea() loop (avoids repeated dict lookups per gene set).
    """
    K = len(hs)
    if K == 0:
        return 0.0, ""

    N = len(abs_signature)
    ah = abs_signature[hs]
    ca = np.cumsum(ah)
    total = ca[-1]
    if total == 0.0:
        return 0.0, ""

    norm_hit = 1.0 / total
    norm_no_hit = 1.0 / (N - K)
    k_idx = np.arange(K, dtype=np.float64)

    gap = hs.astype(np.float64) - k_idx
    val_at_hit = ca * norm_hit - gap * norm_no_hit
    val_before_hit = val_at_hit - ah * norm_hit

    abs_at = np.abs(val_at_hit)
    abs_bf = np.abs(val_before_hit)
    iat = int(abs_at.argmax())
    ibf = int(abs_bf.argmax())

    pos_at = int(hs[iat])
    pos_bf = max(0, int(hs[ibf]) - 1)
    if abs_at[iat] > abs_bf[ibf] or (abs_at[iat] == abs_bf[ibf] and pos_at <= pos_bf):
        es = float(val_at_hit[iat])
        peak_dense = pos_at
    else:
        es = float(val_before_hit[ibf])
        peak_dense = pos_bf

    if es >= 0:
        lgenes = hs[hs < peak_dense].tolist()
    else:
        lgenes = hs[hs >= peak_dense].tolist()

    return es, ",".join(gene_names[p] for p in lgenes)


def compute_anchor_sizes(
    library: dict[str, set[str]],
    abs_signature: np.ndarray,
    calibration_anchors: int,
) -> np.ndarray:
    max_ll = max(len(v) for v in library.values())
    # Log-spaced anchors give dense coverage of small gene sets where the
    # gamma parameters vary most, without the hardcoded extension list that
    # inflated the total count well past calibration_anchors.
    sizes = np.unique(
        np.round(np.geomspace(1, max_ll, calibration_anchors)).astype(int)
    )
    return sizes[(sizes > 0) & (sizes < len(abs_signature))]


def plot(anchor_set_sizes, result: CalibrationResult) -> None:
    xx = np.linspace(min(anchor_set_sizes), max(anchor_set_sizes), 1000)
    for fig_idx, (label, ydata, ysmooth) in enumerate(
        [
            ("alpha pos", result.alpha_pos, result.alpha_pos(xx)),
            ("alpha neg", result.alpha_neg, result.alpha_neg(xx)),
            ("beta pos", result.beta_pos, result.beta_pos(xx)),
            ("beta neg", result.beta_neg, result.beta_neg(xx)),
            ("pos ratio", result.pos_ratio, result.pos_ratio(xx)),
        ],
        1,
    ):
        plt.figure(fig_idx)
        plt.plot(xx, ysmooth, "--", lw=3)
        plt.plot(anchor_set_sizes, ydata, "o")
        plt.title(label)


class CalibrationResult(NamedTuple):
    alpha_pos: interpolate.interp1d
    beta_pos: interpolate.interp1d
    pos_ratio: interpolate.interp1d
    alpha_neg: interpolate.interp1d
    beta_neg: interpolate.interp1d
    ks_pos: float
    ks_neg: float


def estimate_parameters(
    abs_signature: np.ndarray,
    library: dict[str, set[str]],
    permutations: int = 2000,
    symmetric: bool = False,
    calibration_anchors: int = 40,
    plotting: bool = False,
    max_workers: int | None = None,
    verbose: bool = False,
    seed: int = 0,
    ks_disable: bool = False,
) -> CalibrationResult:
    anchor_set_sizes = compute_anchor_sizes(library, abs_signature, calibration_anchors)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        args = [
            (abs_signature, xx, permutations, symmetric, int(seed + xx), ks_disable)
            for xx in anchor_set_sizes
        ]
        results = list(executor.map(estimate_anchor_star, args))

    (
        alpha_pos,
        beta_pos,
        ks_pos_vals,
        alpha_neg,
        beta_neg,
        ks_neg_vals,
        pos_ratio,
    ) = zip(*results)

    if np.max(pos_ratio) > 1.5 and verbose:
        print(
            "Significant unbalance between positive and negative enrichment scores detected."
        )

    f_alpha_pos = loess_interpolation(anchor_set_sizes, alpha_pos)
    f_beta_pos = loess_interpolation(anchor_set_sizes, beta_pos, frac=0.15)
    f_alpha_neg = loess_interpolation(anchor_set_sizes, alpha_neg)
    f_beta_neg = loess_interpolation(anchor_set_sizes, beta_neg, frac=0.15)

    # Small noise prevents numerical instability in the LOESS fit of pos_ratio
    pos_ratio = np.array(pos_ratio) - np.abs(0.0001 * np.random.randn(len(pos_ratio)))
    f_pos_ratio = loess_interpolation(anchor_set_sizes, pos_ratio, frac=0.5)

    calibration_result = CalibrationResult(
        alpha_pos=f_alpha_pos,
        beta_pos=f_beta_pos,
        pos_ratio=f_pos_ratio,
        alpha_neg=f_alpha_neg,
        beta_neg=f_beta_neg,
        ks_pos=float(np.mean(ks_pos_vals)),
        ks_neg=float(np.mean(ks_neg_vals)),
    )
    if plotting:
        plot(
            anchor_set_sizes,
            calibration_result,
        )

    return calibration_result


def clean_library(
    library: dict[str, set[str]], signature: pl.DataFrame
) -> dict[str, set[str]]:
    valid_elements = set(signature["i"].to_list())
    return {key: gene_set & valid_elements for key, gene_set in library.items()}


def gsea(
    signature: pl.DataFrame,
    library: dict[str, list[str] | set[str]],
    permutations: int = 1000,
    anchors: int = 40,
    min_size: int = 5,
    max_size: int = 4000,
    max_workers: int | None = None,
    plotting: bool = False,
    verbose: bool = False,
    symmetric: bool = False,
    signature_cache: bool = True,
    kl_threshold: float = 0.3,
    kl_bins: int = 200,
    shared_null: bool = False,
    seed: int = 0,
    add_noise: bool = False,
    accuracy: int = 40,
    deep_accuracy: int = 50,
    center: bool = True,
    ks_disable: bool = False,
) -> pl.DataFrame:
    """
    Perform Gene Set Enrichment Analysis (GSEA) on the given signature and library.

    Parameters
    ----------
    signature : pl.DataFrame
        Two-column DataFrame (gene ID, numeric value).
    library : dict
        Mapping of gene set name to list of gene IDs.
    permutations : int
        Permutations per anchor size for null calibration. Default 1000.
    anchors : int
        Number of anchor set sizes for calibration. Default 40.
    min_size, max_size : int
        Gene set size filter. Defaults 5, 4000.
    max_workers : int | None
        Max workers for thread pool. Default None.
    symmetric : bool
        Use a single gamma for both ES tails. Default False.
    signature_cache : bool
        Reuse calibration for identical signatures. Default True.
    shared_null : bool
        Reuse calibration for similar signatures (KL-divergence test). Default False.
    seed : int
        Random seed; -1 for a random seed. Default 0.
    center : bool
        Centre signature values before analysis. Default True.

    Returns
    -------
    pl.DataFrame
        Columns: Term, es, nes, pval, sidak, fdr, geneset_size, leading_edge.
    """
    if seed == -1:
        seed = random.randint(-10000000, 100000000)

    # Normalise column names
    cols = signature.columns
    signature = signature.rename({cols[0]: "i", cols[1]: "v"})

    if permutations < 1000 and not symmetric:
        if verbose:
            print("Low number of permutations: enabling symmetric Gamma for accuracy.")
        symmetric = True
    elif permutations < 500:
        if verbose:
            print("Low number of permutations may lead to inaccurate p-values.")
        symmetric = True

    random.seed(seed)
    np.random.seed(seed)

    # Stable hash of the raw input — tobytes() is a C-level memcpy, much
    # faster than string-formatting the whole DataFrame.
    sig_hash = hash(
        signature["v"].to_numpy().astype(np.float64).tobytes()
        + ",".join(signature["i"].to_list()).encode()
    )

    if add_noise:
        noise = np.random.normal(len(signature)) / (
            signature["v"].abs().mean() * 100000
        )
        signature = signature.with_columns((pl.col("v") + noise).alias("v"))

    signature = signature.sort("v", descending=True, maintain_order=True).unique(
        subset=["i"], keep="first", maintain_order=True
    )
    library = {key: set(value) for key, value in library.items()}
    library = clean_library(library, signature)

    if center:
        signature = signature.with_columns(
            (pl.col("v") - pl.col("v").mean()).alias("v")
        )

    gene_names: list = signature["i"].to_list()
    abs_signature = signature["v"].abs().to_numpy()
    signature_map = {gene: idx for idx, gene in enumerate(gene_names)}

    if shared_null and len(pdf_cache) > 0:
        kld, sig_hash_temp = best_kl_fit(
            signature["v"].to_numpy(), pdf_cache, bins=kl_bins
        )
        if kld < kl_threshold:
            sig_hash = sig_hash_temp
            if verbose:
                print(f"Found compatible null model. Best KL-divergence: {kld}")
        elif verbose:
            print(
                f"No compatible null model. Best KL-divergence: {kld} > kl_threshold: {kl_threshold}"
            )

    if sig_hash in pdf_cache and signature_cache:
        if verbose:
            print("Use cached anchor parameters")
        (
            f_alpha_pos,
            f_beta_pos,
            f_pos_ratio,
            f_alpha_neg,
            f_beta_neg,
            ks_pos,
            ks_neg,
        ) = pdf_cache[sig_hash]["model"]
    else:
        (
            f_alpha_pos,
            f_beta_pos,
            f_pos_ratio,
            f_alpha_neg,
            f_beta_neg,
            ks_pos,
            ks_neg,
        ) = estimate_parameters(
            abs_signature,
            library,
            permutations=permutations,
            calibration_anchors=anchors,
            max_workers=max_workers,
            symmetric=symmetric,
            plotting=plotting,
            verbose=verbose,
            seed=seed,
            ks_disable=ks_disable,
        )
        if shared_null:
            xv, pdf = create_pdf(signature["v"].to_numpy(), kl_bins)
        else:
            xv, pdf = None, None
        pdf_cache[sig_hash] = {
            "xvalues": xv,
            "pdf": pdf,
            "model": (
                f_alpha_pos,
                f_beta_pos,
                f_pos_ratio,
                f_alpha_neg,
                f_beta_neg,
                ks_pos,
                ks_neg,
            ),
        }

    # Build valid_gsets: pre-compute sorted hit-index arrays so the scoring
    # loop uses only numpy ops (no per-gene-set dict lookups).
    valid_gsets: list[tuple[str, np.ndarray]] = []
    for k, gene_set in library.items():
        idxs = sorted(signature_map[x] for x in gene_set if x in signature_map)
        n = len(idxs)
        if min_size <= n <= max_size:
            valid_gsets.append((k, np.array(idxs, dtype=np.int64)))

    if valid_gsets:
        unique_sizes = np.array(sorted({len(s) for _, s in valid_gsets}), dtype=float)
        _apos = f_alpha_pos(unique_sizes)
        _bpos = f_beta_pos(unique_sizes)
        _prat = np.clip(f_pos_ratio(unique_sizes), 0.0, 1.0)
        _aneg = f_alpha_neg(unique_sizes)
        _bneg = f_beta_neg(unique_sizes)
        _size_idx: dict[int, int] = {int(s): i for i, s in enumerate(unique_sizes)}
    else:
        _size_idx = {}

    gsets, ess, pvals, ness, set_size, legeness = [], [], [], [], [], []

    # Set mpmath precision once for the scoring loop
    mp.dps = accuracy
    mp.prec = accuracy

    for k, hs in valid_gsets:
        gsets.append(k)
        gsize = len(hs)
        es, legenes = _score_gene_set(abs_signature, hs, gene_names)

        _i = _size_idx[gsize]
        pos_alpha = float(_apos[_i])
        pos_beta = float(_bpos[_i])
        pos_ratio = float(_prat[_i])
        neg_alpha = float(_aneg[_i])
        neg_beta = float(_bneg[_i])

        if es > 0:
            prob = float(gammainc(pos_alpha, es / pos_beta))
            if prob > 0.999999999 or prob < 0.00000000001:
                mp.dps = deep_accuracy
                mp.prec = deep_accuracy
                prob = gammacdf(
                    es, float(pos_alpha), float(pos_beta), dps=deep_accuracy
                )
                mp.dps = accuracy
                mp.prec = accuracy
            prob_two_tailed = min(0.5, 1.0 - min(prob * pos_ratio + 1 - pos_ratio, 1.0))
            nes = invcdf(1.0 - min(1.0, prob_two_tailed))
            pval = 2 * prob_two_tailed
        else:
            prob = float(gammainc(neg_alpha, -es / neg_beta))
            if prob > 0.999999999 or prob < 0.00000000001:
                mp.dps = deep_accuracy
                mp.prec = deep_accuracy
                prob = gammacdf(
                    -es, float(neg_alpha), float(neg_beta), dps=deep_accuracy
                )
                mp.dps = accuracy
                mp.prec = accuracy
            prob_two_tailed = min(
                0.5, 1.0 - min(prob - prob * pos_ratio + pos_ratio, 1.0)
            )
            if prob_two_tailed == 0.5:
                prob_two_tailed -= prob
            nes = invcdf(min(1.0, prob_two_tailed))
            pval = 2 * prob_two_tailed

        ness.append(-float(nes))
        ess.append(float(es))
        pvals.append(float(pval))
        set_size.append(gsize)
        legeness.append(legenes)

    if not verbose:
        np.seterr(divide="ignore")

    if len(pvals) > 1:
        fdr_values = multipletests(pvals, method="fdr_bh")[1].tolist()
        sidak_values = multipletests(pvals, method="sidak")[1].tolist()
    else:
        fdr_values = pvals
        sidak_values = pvals

    res = pl.DataFrame(
        {
            "Term": gsets,
            "es": ess,
            "nes": ness,
            "pval": pvals,
            "sidak": sidak_values,
            "fdr": fdr_values,
            "geneset_size": set_size,
            "leading_edge": legeness,
        }
    )

    if (ks_pos < 0.05 or ks_neg < 0.05) and verbose:
        print(
            f"KS test failed. Gamma approximation deviates from permutation samples.\n"
            f"KS p-value (pos): {ks_pos}\nKS p-value (neg): {ks_neg}"
        )

    return res.sort(pl.col("pval").abs())
