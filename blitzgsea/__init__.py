import random
from concurrent.futures import ThreadPoolExecutor

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
    es = np.array(get_peak_size_adv(abs_signature, set_size, permutations, int(seed)))

    pos = es[es > 0]
    neg = es[es < 0]

    if (len(neg) < 250 or len(pos) < 250) and not symmetric:
        symmetric = True

    if symmetric:
        aes = np.abs(es)[es != 0]
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

        fit_alpha, fit_loc, fit_beta = gamma.fit(-np.array(neg), floc=0)
        if not ks_disable:
            ks_neg = kstest(
                -np.array(neg), "gamma", args=(fit_alpha, fit_loc, fit_beta)
            )[1]
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
) -> list[float]:
    """
    Generate null-distribution ES values via O(K) sparse sampling.

    Uses exponential-spacing order statistics to place K hits in O(K) per
    permutation instead of O(N), giving a ~300x speedup for typical N >> K.
    Hit positions are sorted by construction, so the running sum can be
    evaluated analytically at only 2K candidate extrema rather than by a
    dense cumsum over N elements.
    """
    rng = np.random.default_rng(seed)
    N = len(abs_signature)
    K = number_hits
    number_miss = N - K
    norm_no_hit = np.float32(1.0 / number_miss)
    abs_sig_f32 = abs_signature.astype(np.float32)
    k_idx = np.arange(K, dtype=np.float32)  # 0..K-1 for gap computation

    # ~6 float32 arrays of size (batch, K) → 24 bytes per element
    batch_size = max(1, min(permutations, int(50_000_000 // (K * 24))))

    es_chunks: list = []
    remaining = permutations

    while remaining > 0:
        batch = min(batch_size, remaining)
        remaining -= batch

        # O(K) sorted without-replacement sampling via exponential spacings.
        # K+1 i.i.d. Exp(1) values; their normalised prefix cumsums are the
        # order statistics of K Uniform[0,1] draws (Devroye 1986).
        exp = rng.exponential(1.0, size=(batch, K + 1)).astype(np.float32)
        cumexp = np.cumsum(exp, axis=1)
        U = cumexp[:, :K] / cumexp[:, -1:]
        hs = np.clip(np.floor(U * N).astype(np.int32), 0, N - 1)  # (batch, K)

        # Resample any rows with collisions (~6% at typical K).
        for _ in range(8):
            bad = (np.diff(hs, axis=1) == 0).any(axis=1)
            if not bad.any():
                break
            nb = int(bad.sum())
            e2 = rng.exponential(1.0, size=(nb, K + 1)).astype(np.float32)
            c2 = np.cumsum(e2, axis=1)
            hs[bad] = np.clip(
                np.floor((c2[:, :K] / c2[:, -1:]) * N).astype(np.int32), 0, N - 1
            )

        # O(K) sparse ES formula.
        # hs is sorted ascending; ah[b,k] = abs_signature[hs[b,k]].
        ah = abs_sig_f32[hs]  # (batch, K) — hit absolute values
        ca = np.cumsum(ah, axis=1)  # (batch, K) — cumulative hit weight up to k-th hit
        nh = np.float32(1.0) / ca[:, -1:]  # (batch, 1) — 1 / total hit weight

        # gap[b,k] = number of miss positions before the k-th hit = hs[b,k] - k
        gap = hs.astype(np.float32) - k_idx  # (batch, K)
        val_at_hit = ca * nh - gap * norm_no_hit  # running sum after k-th hit
        val_before_hit = val_at_hit - ah * nh  # running sum just before k-th hit

        # Peak ES = candidate with largest absolute running sum across 2K points
        abs_at = np.abs(val_at_hit)
        abs_bf = np.abs(val_before_hit)
        r = np.arange(batch)
        iat = abs_at.argmax(axis=1)
        ibf = abs_bf.argmax(axis=1)
        es_batch = np.where(
            abs_at[r, iat] >= abs_bf[r, ibf],
            val_at_hit[r, iat],
            val_before_hit[r, ibf],
        )

        valid = ~np.isnan(es_batch)
        es_chunks.append(es_batch[valid])

    return np.concatenate(es_chunks).tolist()


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
    signature_map: dict[str, int],
    gene_set: list[str] | set[str],
    gene_names: list[str],
) -> tuple[float, str]:
    """O(K) enrichment score and leading edge for a single gene set.

    Replaces the enrichment_score() + get_leading_edge() pair in the main
    scoring loop.  Uses the same sparse running-sum formula as
    get_peak_size_adv(), avoiding O(N) cumsum and O(N) argmax.
    """
    hits_sorted = sorted(signature_map[x] for x in gene_set if x in signature_map)
    K = len(hits_sorted)
    if K == 0:
        return 0.0, ""

    N = len(abs_signature)
    hs = np.array(hits_sorted, dtype=np.int64)
    ah = abs_signature[hs]  # hit absolute values (float64 for accuracy)
    ca = np.cumsum(ah)  # cumulative hit weight
    total = ca[-1]
    if total == 0.0:
        return 0.0, ""

    norm_hit = 1.0 / total
    norm_no_hit = 1.0 / (N - K)
    k_idx = np.arange(K, dtype=np.float64)

    gap = hs.astype(np.float64) - k_idx  # miss positions before each hit
    val_at_hit = ca * norm_hit - gap * norm_no_hit
    val_before_hit = val_at_hit - ah * norm_hit

    abs_at = np.abs(val_at_hit)
    abs_bf = np.abs(val_before_hit)
    iat = int(abs_at.argmax())
    ibf = int(abs_bf.argmax())

    if abs_at[iat] >= abs_bf[ibf]:
        es = float(val_at_hit[iat])
        peak_dense = int(hs[iat])
    else:
        es = float(val_before_hit[ibf])
        peak_dense = max(0, int(hs[ibf]) - 1)

    if es >= 0:
        lgenes = [p for p in hits_sorted if p < peak_dense]
    else:
        lgenes = [p for p in hits_sorted if p >= peak_dense]

    return es, ",".join(gene_names[p] for p in lgenes)


def estimate_parameters(
    abs_signature: np.ndarray,
    library: dict[str, set[str]],
    permutations: int = 2000,
    max_size: int = 4000,
    symmetric: bool = False,
    calibration_anchors: int = 40,
    plotting: bool = False,
    processes: int = 4,
    verbose: bool = False,
    progress: bool = False,
    seed: int = 0,
    ks_disable: bool = False,
) -> tuple[
    interpolate.interp1d,
    interpolate.interp1d,
    interpolate.interp1d,
    interpolate.interp1d,
    interpolate.interp1d,
    float,
    float,
]:
    max_ll = int(np.max([len(v) for v in library.values()]))

    # Log-spaced anchors give dense coverage of small gene sets where the
    # gamma parameters vary most, without the hardcoded extension list that
    # inflated the total count well past calibration_anchors.
    anchor_set_sizes = np.unique(
        np.round(np.geomspace(1, max_ll, calibration_anchors)).astype(int)
    ).tolist()
    anchor_set_sizes = [s for s in anchor_set_sizes if 0 < s < len(abs_signature)]

    if processes == 1:
        results = list(
            estimate_anchor(
                abs_signature, xx, permutations, symmetric, int(seed + xx), ks_disable
            )
            for xx in anchor_set_sizes
        )
    else:
        with ThreadPoolExecutor(max_workers=processes) as executor:
            args = [
                (abs_signature, xx, permutations, symmetric, int(seed + xx), ks_disable)
                for xx in anchor_set_sizes
            ]
            results = list(executor.map(estimate_anchor_star, args))

    alpha_pos, beta_pos, ks_pos_vals = [], [], []
    alpha_neg, beta_neg, ks_neg_vals = [], [], []
    pos_ratio = []

    for (
        f_alpha_pos,
        f_beta_pos,
        f_ks_pos,
        f_alpha_neg,
        f_beta_neg,
        f_ks_neg,
        f_pos_ratio,
    ) in results:
        alpha_pos.append(f_alpha_pos)
        beta_pos.append(f_beta_pos)
        ks_pos_vals.append(f_ks_pos)
        alpha_neg.append(f_alpha_neg)
        beta_neg.append(f_beta_neg)
        ks_neg_vals.append(f_ks_neg)
        pos_ratio.append(f_pos_ratio)

    if np.max(pos_ratio) > 1.5 and verbose:
        print(
            "Significant unbalance between positive and negative enrichment scores detected."
        )

    anchor_set_sizes = np.array(anchor_set_sizes, dtype=float)

    f_alpha_pos = loess_interpolation(anchor_set_sizes, alpha_pos)
    f_beta_pos = loess_interpolation(anchor_set_sizes, beta_pos, frac=0.15)
    f_alpha_neg = loess_interpolation(anchor_set_sizes, alpha_neg)
    f_beta_neg = loess_interpolation(anchor_set_sizes, beta_neg, frac=0.15)

    # Small noise prevents numerical instability in the LOESS fit of pos_ratio
    pos_ratio = np.array(pos_ratio) - np.abs(0.0001 * np.random.randn(len(pos_ratio)))
    f_pos_ratio = loess_interpolation(anchor_set_sizes, pos_ratio, frac=0.5)

    if plotting:
        xx = np.linspace(min(anchor_set_sizes), max(anchor_set_sizes), 1000)
        for fig_idx, (label, ydata, ysmooth) in enumerate(
            [
                ("alpha pos", alpha_pos, f_alpha_pos(xx)),
                ("alpha neg", alpha_neg, f_alpha_neg(xx)),
                ("beta pos", beta_pos, f_beta_pos(xx)),
                ("beta neg", beta_neg, f_beta_neg(xx)),
                ("pos ratio", pos_ratio, f_pos_ratio(xx)),
            ],
            1,
        ):
            plt.figure(fig_idx)
            plt.plot(xx, ysmooth, "--", lw=3)
            plt.plot(anchor_set_sizes, ydata, "o")
            plt.title(label)

    return (
        f_alpha_pos,
        f_beta_pos,
        f_pos_ratio,
        f_alpha_neg,
        f_beta_neg,
        np.mean(ks_pos_vals),
        np.mean(ks_neg_vals),
    )


def clean_library(
    library: dict[str, set[str]], signature: pl.DataFrame
) -> dict[str, set[str]]:
    valid_elements = set(signature["i"].to_list())
    return {key: gene_set & valid_elements for key, gene_set in library.items()}


def gsea(
    signature: pl.DataFrame,
    library: dict[str, list[str]],
    permutations: int = 1000,
    anchors: int = 40,
    min_size: int = 5,
    max_size: int = 4000,
    processes: int = 4,
    plotting: bool = False,
    verbose: bool = False,
    progress: bool = False,
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
    processes : int
        Parallel calibration workers. Default 4.
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

    # Sort descending, deduplicate on gene ID keeping first (highest-ranked).
    signature = signature.sort("v", descending=True).unique(
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
            processes=processes,
            symmetric=symmetric,
            plotting=plotting,
            verbose=verbose,
            seed=seed,
            progress=progress,
            max_size=max_size,
            ks_disable=ks_disable,
        )
        xv, pdf = create_pdf(signature["v"].to_numpy(), kl_bins)
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

    signature_genes = set(gene_names)

    # First pass: collect valid sets so we can batch-evaluate the 5 LOESS
    # interpolators once per unique size rather than once per gene set.
    valid_gsets: list[tuple[str, list[str]]] = []
    for k in library.keys():
        stripped = strip_gene_set(signature_genes, library[k])
        if min_size <= len(stripped) <= max_size:
            valid_gsets.append((k, stripped))

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

    for k, stripped_set in valid_gsets:
        gsets.append(k)
        gsize = len(stripped_set)
        es, legenes = _score_gene_set(
            abs_signature, signature_map, stripped_set, gene_names
        )

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
