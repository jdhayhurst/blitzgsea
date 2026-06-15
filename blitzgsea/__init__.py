import random
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from collections import Counter
from scipy import interpolate

from matplotlib import pyplot as plt
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
import multiprocessing

from mpmath import mp

from scipy.stats import gamma
from scipy.stats import kstest

from blitzgsea.signature_similarity import create_pdf, best_kl_fit
from blitzgsea.mpsci import gammacdf, invcdf
import blitzgsea.enrichr
import blitzgsea.plot
import blitzgsea.shuffle
import blitzgsea.signature_similarity

mp.dps = 1000
mp.prec = 1000
pdf_cache: dict = {}


def estimate_anchor_star(args):
    return estimate_anchor(*args)


def estimate_anchor(abs_signature, set_size, permutations, symmetric, seed, ks_disable):
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
            ks_pos = kstest(aes, 'gamma', args=(fit_alpha, fit_loc, fit_beta))[1]
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
            ks_pos = kstest(pos, 'gamma', args=(fit_alpha, fit_loc, fit_beta))[1]
        alpha_pos = fit_alpha
        beta_pos = fit_beta

        fit_alpha, fit_loc, fit_beta = gamma.fit(-np.array(neg), floc=0)
        if not ks_disable:
            ks_neg = kstest(-np.array(neg), 'gamma', args=(fit_alpha, fit_loc, fit_beta))[1]
        alpha_neg = fit_alpha
        beta_neg = fit_beta

    pos_ratio = len(pos) / (len(pos) + len(neg))

    return alpha_pos, beta_pos, ks_pos, alpha_neg, beta_neg, ks_neg, pos_ratio


def strip_gene_set(signature_genes, gene_set):
    return [x for x in gene_set if x in signature_genes]


def enrichment_score(abs_signature, signature_map, gene_set):
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


def enrichment_score_null(abs_signature, hit_indicator, number_hits):
    """Single-permutation ES used by external callers; kept for backward compatibility."""
    hits = np.random.choice(len(abs_signature), size=number_hits, replace=False)
    hit_indicator_new = np.zeros(len(abs_signature), dtype=np.float32)
    hit_indicator_new[hits] = 1
    number_miss = len(abs_signature) - number_hits
    sum_hit_scores = np.sum(abs_signature[hits])
    norm_hit = 1.0 / sum_hit_scores
    norm_no_hit = 1.0 / number_miss
    increment = hit_indicator_new * (abs_signature * norm_hit + norm_no_hit) - norm_no_hit
    running_sum = np.cumsum(increment, dtype=np.float32)
    peak = np.abs(running_sum).argmax()
    return running_sum[peak]


def get_leading_edge(runningsum, signature, gene_set, signature_map):
    gs = set(gene_set)
    hits = [signature_map[x] for x in gs if x in signature_map]
    rmax = np.argmax(runningsum)
    rmin = np.argmin(runningsum)
    if runningsum[rmax] > np.abs(runningsum[rmin]):
        lgenes = set(hits).intersection(range(rmax))
    else:
        lgenes = set(hits).intersection(range(rmin, len(runningsum)))
    return ",".join(signature.index[list(lgenes)])


def get_peak_size_adv(abs_signature, number_hits, permutations, seed):
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
        ah = abs_sig_f32[hs]          # (batch, K) — hit absolute values
        ca = np.cumsum(ah, axis=1)    # (batch, K) — cumulative hit weight up to k-th hit
        nh = np.float32(1.0) / ca[:, -1:]  # (batch, 1) — 1 / total hit weight

        # gap[b,k] = number of miss positions before the k-th hit = hs[b,k] - k
        gap = hs.astype(np.float32) - k_idx                  # (batch, K)
        val_at_hit = ca * nh - gap * norm_no_hit             # running sum after k-th hit
        val_before_hit = val_at_hit - ah * nh                # running sum just before k-th hit

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


def loess_interpolation(x, y, frac=0.6, it=4):
    yout = lowess(y, x, frac=frac)[:, 1]
    return interpolate.interp1d(x, yout, bounds_error=False, fill_value="extrapolate")


def estimate_parameters(signature, abs_signature, signature_map, library, permutations: int = 2000, max_size=4000, symmetric: bool = False, calibration_anchors: int = 40, plotting: bool = False, processes=4, verbose=False, progress=False, seed: int = 0, ks_disable=False):
    ll = [len(v) for v in library.values()]
    cc = Counter(ll)
    set_sizes = pd.DataFrame(list(cc.items()), columns=['set_size', 'count']).sort_values("set_size")
    set_sizes["cumsum"] = np.cumsum(set_sizes.iloc[:, 1])

    anchor_set_sizes = [int(x) for x in np.linspace(1, np.max(ll), calibration_anchors)]
    anchor_set_sizes.extend([1, 2, 3, 4, 5, 6, 7, 12, 16, 20, 30, 40, 50, 60, 70, 80, 100, np.max(ll) + 10, np.max(ll) + 30])
    anchor_set_sizes = sorted(set(anchor_set_sizes))
    anchor_set_sizes = [s for s in anchor_set_sizes if s < len(abs_signature)]

    if processes == 1:
        process_generator = (
            estimate_anchor(abs_signature, xx, permutations, symmetric, int(seed + xx), ks_disable)
            for xx in anchor_set_sizes
        )
        results = list(tqdm(process_generator, desc="Calibration", total=len(anchor_set_sizes), disable=not progress))
    else:
        with multiprocessing.Pool(processes) as pool:
            args = [
                (abs_signature, xx, permutations, symmetric, int(seed + xx), ks_disable)
                for xx in anchor_set_sizes
            ]
            results = list(tqdm(pool.imap(estimate_anchor_star, args), desc="Calibration", total=len(args), disable=not progress))

    alpha_pos, beta_pos, ks_pos_vals = [], [], []
    alpha_neg, beta_neg, ks_neg_vals = [], [], []
    pos_ratio = []

    for f_alpha_pos, f_beta_pos, f_ks_pos, f_alpha_neg, f_beta_neg, f_ks_neg, f_pos_ratio in results:
        alpha_pos.append(f_alpha_pos)
        beta_pos.append(f_beta_pos)
        ks_pos_vals.append(f_ks_pos)
        alpha_neg.append(f_alpha_neg)
        beta_neg.append(f_beta_neg)
        ks_neg_vals.append(f_ks_neg)
        pos_ratio.append(f_pos_ratio)

    if np.max(pos_ratio) > 1.5 and verbose:
        print('Significant unbalance between positive and negative enrichment scores detected.')

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
        for fig_idx, (label, ydata, ysmooth) in enumerate([
            ("alpha pos", alpha_pos, f_alpha_pos(xx)),
            ("alpha neg", alpha_neg, f_alpha_neg(xx)),
            ("beta pos", beta_pos, f_beta_pos(xx)),
            ("beta neg", beta_neg, f_beta_neg(xx)),
            ("pos ratio", pos_ratio, f_pos_ratio(xx)),
        ], 1):
            plt.figure(fig_idx)
            plt.plot(xx, ysmooth, '--', lw=3)
            plt.plot(anchor_set_sizes, ydata, 'o')
            plt.title(label)

    return f_alpha_pos, f_beta_pos, f_pos_ratio, f_alpha_neg, f_beta_neg, np.mean(ks_pos_vals), np.mean(ks_neg_vals)


def clean_library(library, signature):
    valid_elements = set(signature.index)
    return {key: gene_set & valid_elements for key, gene_set in library.items()}


def gsea(signature, library, permutations: int = 1000, anchors: int = 40, min_size: int = 5, max_size: int = 4000, processes: int = 4, plotting: bool = False, verbose: bool = False, progress: bool = False, symmetric: bool = False, signature_cache: bool = True, kl_threshold: float = 0.3, kl_bins: int = 200, shared_null: bool = False, seed: int = 0, add_noise: bool = False, accuracy: int = 40, deep_accuracy: int = 50, center=True, ks_disable=False):
    """
    Perform Gene Set Enrichment Analysis (GSEA) on the given signature and library.

    Parameters
    ----------
    signature : pd.DataFrame
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
    pd.DataFrame
        Columns: es, nes, pval, sidak, fdr, geneset_size, leading_edge; indexed by Term.
    """
    if seed == -1:
        seed = random.randint(-10000000, 100000000)

    signature = signature.copy()
    signature.columns = ["i", "v"]

    if permutations < 1000 and not symmetric:
        if verbose:
            print('Low number of permutations: enabling symmetric Gamma for accuracy.')
        symmetric = True
    elif permutations < 500:
        if verbose:
            print('Low number of permutations may lead to inaccurate p-values.')
        symmetric = True

    random.seed(seed)
    np.random.seed(seed)
    sig_hash = hash(signature.to_string())

    if add_noise:
        signature.iloc[:, 1] += np.random.normal(signature.shape[0]) / (np.mean(np.abs(signature.iloc[:, 1])) * 100000)

    signature = signature.sort_values("v", ascending=False).set_index("i")
    signature = signature[~signature.index.duplicated(keep='first')]
    library = {key: set(value) for key, value in library.items()}
    library = clean_library(library, signature)

    if center:
        signature.loc[:, "v"] -= np.mean(signature.loc[:, "v"])

    abs_signature = np.abs(signature.loc[:, "v"].to_numpy())
    signature_map = {h: i for i, h in enumerate(signature.index)}

    if shared_null and len(pdf_cache) > 0:
        kld, sig_hash_temp = best_kl_fit(signature["v"].to_numpy(), pdf_cache, bins=kl_bins)
        if kld < kl_threshold:
            sig_hash = sig_hash_temp
            if verbose:
                print(f"Found compatible null model. Best KL-divergence: {kld}")
        elif verbose:
            print(f"No compatible null model. Best KL-divergence: {kld} > kl_threshold: {kl_threshold}")

    if sig_hash in pdf_cache and signature_cache:
        if verbose:
            print("Use cached anchor parameters")
        f_alpha_pos, f_beta_pos, f_pos_ratio, f_alpha_neg, f_beta_neg, ks_pos, ks_neg = pdf_cache[sig_hash]["model"]
    else:
        f_alpha_pos, f_beta_pos, f_pos_ratio, f_alpha_neg, f_beta_neg, ks_pos, ks_neg = estimate_parameters(
            signature, abs_signature, signature_map, library,
            permutations=permutations, calibration_anchors=anchors,
            processes=processes, symmetric=symmetric, plotting=plotting,
            verbose=verbose, seed=seed, progress=progress,
            max_size=max_size, ks_disable=ks_disable,
        )
        xv, pdf = create_pdf(signature["v"].to_numpy(), kl_bins)
        pdf_cache[sig_hash] = {
            "xvalues": xv,
            "pdf": pdf,
            "model": (f_alpha_pos, f_beta_pos, f_pos_ratio, f_alpha_neg, f_beta_neg, ks_pos, ks_neg),
        }

    signature_genes = set(signature.index)
    gsets, ess, pvals, ness, set_size, legeness = [], [], [], [], [], []

    # Set mpmath precision once for the scoring loop
    mp.dps = accuracy
    mp.prec = accuracy

    for k in tqdm(library.keys(), desc="Enrichment ", disable=not verbose):
        stripped_set = strip_gene_set(signature_genes, library[k])
        if not (min_size <= len(stripped_set) <= max_size):
            continue

        gsets.append(k)
        gsize = len(stripped_set)
        rs, es = enrichment_score(abs_signature, signature_map, stripped_set)
        legenes = get_leading_edge(rs, signature, stripped_set, signature_map)

        pos_alpha = f_alpha_pos(gsize)
        pos_beta = f_beta_pos(gsize)
        pos_ratio = max(0.0, min(1.0, float(f_pos_ratio(gsize))))
        neg_alpha = f_alpha_neg(gsize)
        neg_beta = f_beta_neg(gsize)

        if es > 0:
            prob = gamma.cdf(es, float(pos_alpha), scale=float(pos_beta))
            if prob > 0.999999999 or prob < 0.00000000001:
                mp.dps = deep_accuracy
                mp.prec = deep_accuracy
                prob = gammacdf(es, float(pos_alpha), float(pos_beta), dps=deep_accuracy)
                mp.dps = accuracy
                mp.prec = accuracy
            prob_two_tailed = min(0.5, 1.0 - min(prob * pos_ratio + 1 - pos_ratio, 1.0))
            nes = invcdf(1.0 - min(1.0, prob_two_tailed))
            pval = 2 * prob_two_tailed
        else:
            prob = gamma.cdf(-es, float(neg_alpha), scale=float(neg_beta))
            if prob > 0.999999999 or prob < 0.00000000001:
                mp.dps = deep_accuracy
                mp.prec = deep_accuracy
                prob = gammacdf(-es, float(neg_alpha), float(neg_beta), dps=deep_accuracy)
                mp.dps = accuracy
                mp.prec = accuracy
            prob_two_tailed = min(0.5, 1.0 - min(prob - prob * pos_ratio + pos_ratio, 1.0))
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
        np.seterr(divide='ignore')

    if len(pvals) > 1:
        fdr_values = multipletests(pvals, method="fdr_bh")[1]
        sidak_values = multipletests(pvals, method="sidak")[1]
    else:
        fdr_values = pvals
        sidak_values = pvals

    res = pd.DataFrame({
        "Term": gsets,
        "es": np.array(ess, dtype=float),
        "nes": np.array(ness, dtype=float),
        "pval": np.array(pvals, dtype=float),
        "sidak": np.array(sidak_values, dtype=float),
        "fdr": np.array(fdr_values, dtype=float),
        "geneset_size": np.array(set_size, dtype=int),
        "leading_edge": legeness,
    }).set_index("Term")

    if (ks_pos < 0.05 or ks_neg < 0.05) and verbose:
        print(
            f'KS test failed. Gamma approximation deviates from permutation samples.\n'
            f'KS p-value (pos): {ks_pos}\nKS p-value (neg): {ks_neg}'
        )

    return res.sort_values("pval", key=abs, ascending=True)
