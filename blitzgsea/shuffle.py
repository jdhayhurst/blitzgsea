import numpy as np
import polars as pl
from statsmodels.stats.multitest import multipletests


def gsea(
    exprs: pl.DataFrame,
    library: dict[str, list[str]],
    groups: list[int] | np.ndarray,
    permutations: int = 1000,
    seed: int = 1,
) -> pl.DataFrame:
    """Phenotype-permutation GSEA.

    Parameters
    ----------
    exprs : pl.DataFrame
        Rows = genes.  First column holds gene IDs; remaining columns are
        per-sample expression values.
    library : dict
        Gene set name → list of gene IDs.
    groups : array-like of int
        Sample group labels (0 = positive class, 1 = negative class).
    """
    gene_col = exprs.columns[0]
    genes = np.array(exprs[gene_col].to_list())
    expr_matrix = exprs.select(pl.all().exclude(gene_col)).to_numpy()  # (N_genes, N_samples)
    expr_mat = expr_matrix.T                                            # (N_samples, N_genes)

    rs = np.random.RandomState(seed)
    perm_cor_tensor = np.tile(expr_mat, (permutations, 1, 1))
    for arr in perm_cor_tensor[:-1]:
        rs.shuffle(arr)

    groups = np.array(groups)
    pos_mask = groups == 0
    neg_mask = groups == 1
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())

    pos_mean = perm_cor_tensor[:, pos_mask, :].mean(axis=1)
    neg_mean = perm_cor_tensor[:, neg_mask, :].mean(axis=1)
    pos_std = perm_cor_tensor[:, pos_mask, :].std(axis=1, ddof=1)
    neg_std = perm_cor_tensor[:, neg_mask, :].std(axis=1, ddof=1)

    denom = np.sqrt(pos_std ** 2 / n_pos + neg_std ** 2 / n_neg)
    cor_mat = (pos_mean - neg_mean) / denom

    gene_mat = cor_mat.argsort()[:, ::-1]
    cor_mat = cor_mat[:, ::-1].T

    keys = np.array(list(library.keys()))
    tag_indicator = np.vstack([np.in1d(genes, library[key], assume_unique=True) for key in keys]).astype(int)
    perm_tag_tensor = np.stack([tag.take(gene_mat).T for tag in tag_indicator], axis=0)

    no_tag_tensor = 1 - perm_tag_tensor
    rank_alpha = np.abs(perm_tag_tensor * cor_mat[np.newaxis, :, :])

    P_GW = rank_alpha.sum(axis=1, keepdims=True)
    P_NG = no_tag_tensor.sum(axis=1, keepdims=True)
    RES_tensor = np.cumsum(rank_alpha / P_GW - no_tag_tensor / P_NG, axis=1)

    es_max = RES_tensor.max(axis=1)
    es_min = RES_tensor.min(axis=1)
    es_matrix = np.where(np.abs(es_max) > np.abs(es_min), es_max, es_min)

    es = es_matrix[:, -1]
    es_null = es_matrix[:, :-1]

    pvals = np.where(
        es < 0,
        (es_null < es.reshape(-1, 1)).sum(axis=1) / (es_null < 0).sum(axis=1),
        (es_null >= es.reshape(-1, 1)).sum(axis=1) / (es_null >= 0).sum(axis=1),
    )
    fdr_values = multipletests(pvals, method="fdr_bh")[1]
    sidak_values = multipletests(pvals, method="sidak")[1]

    return pl.DataFrame({
        "Term": list(keys),
        "es": es.astype(float).tolist(),
        "pval": pvals.astype(float).tolist(),
        "fdr": fdr_values.astype(float).tolist(),
        "sidak": sidak_values.astype(float).tolist(),
    }).sort("pval")
