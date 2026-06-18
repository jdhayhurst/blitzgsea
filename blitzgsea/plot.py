import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import blitzgsea as blitz


def _prepare_signature(signature: pl.DataFrame, center: bool) -> pl.DataFrame:
    """Sort, dedup, optionally center; return polars DataFrame with cols i, v."""
    cols = signature.columns
    sig = signature.rename({cols[0]: "i", cols[1]: "v"})
    sig = sig.sort("v", descending=True).unique(
        subset=["i"], keep="first", maintain_order=True
    )
    if center:
        sig = sig.with_columns((pl.col("v") - pl.col("v").mean()).alias("v"))
    return sig


def _clean_library(
    library: dict[str, set[str]],
    signature: pl.DataFrame,
) -> dict[str, set[str]]:
    valid = set(signature["i"].to_list())
    return {key: gene_set & valid for key, gene_set in library.items()}


def running_sum(
    signature: pl.DataFrame,
    geneset: str,
    library: dict[str, list[str] | set[str]],
    result: pl.DataFrame | None = None,
    compact: bool = False,
    center: bool = True,
    interactive_plot: bool = False,
) -> Figure:
    """Plot the enrichment running sum for one gene set."""
    if not interactive_plot:
        plt.ioff()

    sig = _prepare_signature(signature, center)
    library = {key: set(value) for key, value in library.items()}
    library = _clean_library(library, sig)

    gene_list = sig["i"].to_list()
    gs = set(library[geneset])
    hits = [i for i, x in enumerate(gene_list) if x in gs]

    abs_sig = sig["v"].abs().to_numpy()
    sig_map = {gene: idx for idx, gene in enumerate(gene_list)}
    running_sum_arr, es = blitz.enrichment_score(abs_sig, sig_map, gs)
    running_sum_list = running_sum_arr.tolist()

    fig = plt.figure(figsize=(7, 5))

    if compact:
        ax = fig.add_gridspec(5, 11, wspace=0, hspace=0)
        ax1 = fig.add_subplot(ax[0:4, 0:11])
    else:
        ax = fig.add_gridspec(12, 11, wspace=0, hspace=0)
        ax1 = fig.add_subplot(ax[0:7, 0:11])

    lw = 5 if compact else 3
    fs_tick = 24 if compact else 16
    ax1.plot(running_sum_list, color=(0, 1, 0), lw=lw)
    ax1.tick_params(labelsize=fs_tick)
    plt.xlim([0, len(running_sum_list)])

    nn = int(np.abs(running_sum_arr).argmax())
    ax1.vlines(
        x=nn,
        ymin=np.min(running_sum_list),
        ymax=np.max(running_sum_list),
        linestyle=":",
        color="red",
    )

    fs_label = 25 if compact else 20
    va = "bottom" if es > 0 else "top"
    if result is not None:
        nes_val = result.filter(pl.col("Term") == geneset)["nes"][0]
        label = f"NES={nes_val:.3f}"
    else:
        label = f"ES={running_sum_list[nn]:.3f}"
    ax1.text(
        len(running_sum_list) / 30,
        0,
        label,
        size=fs_label,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1},
        ha="left",
        va=va,
        zorder=100,
    )

    ax1.grid(True, which="both")
    ax1.set(xticks=[])
    plt.title(geneset, fontsize=18)
    plt.ylabel("ES" if compact else "Enrichment Score (ES)", fontsize=fs_tick)

    rank_vec = sig["v"].to_numpy()
    N = len(rank_vec)

    if compact:
        ax1 = fig.add_subplot(ax[4:, 0:11])
        ax1.vlines(x=hits, ymin=0, ymax=1, color=(0, 0, 0, 1), lw=1.5)
        plt.xlim([0, N])
        plt.ylim([0, 1])
        ax1.set(yticks=[], xticks=[])
        posv = np.percentile(
            range(len(rank_vec[rank_vec > 0])), np.linspace(0, 100, 10)
        )
        for i in range(9):
            plt.gca().add_patch(
                Rectangle(
                    (posv[i], 0),
                    posv[i + 1] - posv[i],
                    0.5,
                    linewidth=0,
                    facecolor="red",
                    alpha=0.6 * (1 - i * 0.1),
                )
            )
        negv = np.percentile(
            range(len(rank_vec[rank_vec <= 0])), np.linspace(0, 100, 10)
        )
        for i in range(9):
            plt.gca().add_patch(
                Rectangle(
                    (posv[-1] + negv[i], 0),
                    negv[i + 1] - negv[i],
                    0.5,
                    linewidth=0,
                    facecolor="blue",
                    alpha=0.6 * (0.1 + i * 0.1),
                )
            )
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.xlabel("Rank", fontsize=24)
    else:
        ax1 = fig.add_subplot(ax[7:8, 0:11])
        ax1.vlines(x=hits, ymin=-1, ymax=1, color=(0, 0, 0, 1), lw=0.5)
        plt.xlim([0, N])
        plt.ylim([-1, 1])
        ax1.set(yticks=[], xticks=[])

        ax1 = fig.add_subplot(ax[8:, 0:11])
        x = np.append(np.arange(0, N, 20, dtype=int), N - 1)
        ax1.fill_between(x, rank_vec[x], color="lightgrey")
        ax1.plot(x, rank_vec[x], color=(0.2, 0.2, 0.2), lw=1)
        ax1.hlines(y=0, xmin=0, xmax=N, color="black", zorder=100, lw=0.6)
        plt.xlim([0, N])
        plt.ylim([rank_vec.min(), rank_vec.max()])
        minabs = np.abs(rank_vec).min()
        zero_cross = int(np.where(np.abs(rank_vec) == minabs)[0][0])
        ax1.vlines(
            x=zero_cross, ymin=rank_vec.min(), ymax=rank_vec.max(), linestyle=":"
        )
        ax1.text(
            zero_cross,
            rank_vec.max() / 3,
            f"Zero crosses at {zero_cross}",
            bbox={"facecolor": "white", "alpha": 0.5, "edgecolor": "none", "pad": 1},
            ha="center",
            va="center",
        )
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.xlabel("Rank in Ordered Dataset", fontsize=16)
        plt.ylabel("Ranked list metric", fontsize=16)
        ax1.tick_params(labelsize=16)

    if not interactive_plot:
        plt.ion()
    fig.patch.set_facecolor("white")
    return fig


def top_table(
    signature: pl.DataFrame,
    library: dict[str, list[str] | set[str]],
    result: pl.DataFrame,
    n: int = 10,
    center: bool = True,
    interactive_plot: bool = False,
) -> Figure:
    """Plot a summary table of the top N enriched gene sets."""
    if not interactive_plot:
        plt.ioff()

    sig = _prepare_signature(signature, center)
    library = {key: set(value) for key, value in library.items()}
    library = _clean_library(library, sig)

    gene_list = sig["i"].to_list()
    N = len(gene_list)

    fig = plt.figure(figsize=(5, 0.5 * n), frameon=False)
    ax = fig.add_subplot(111)
    fig.patch.set_visible(False)
    plt.axis("off")

    ax.vlines(x=[0.2, 0.8], ymin=-0.1, ymax=1, color="black")
    ln = np.linspace(-0.1, 1, n + 1)[::-1]
    ax.hlines(y=ln, xmin=0, xmax=1, color="black")

    ax.text(0.03, 1.03, "NES", fontsize=16)
    ax.text(0.84, 1.03, "SET", fontsize=16)

    terms = result["Term"].to_list()
    nes_vals = result["nes"].to_list()

    for i in range(n):
        term = terms[i]
        nes = nes_vals[i]
        ax.text(0.03, (ln[i] + ln[i + 1]) / 2, f"{nes:.3f}", verticalalignment="center")
        ax.text(0.84, (ln[i] + ln[i + 1]) / 2, term, verticalalignment="center")

        gs = set(library[term])
        hits = np.array([j for j, x in enumerate(gene_list) if x in gs])
        hits = (hits / N) * 0.6 + 0.2
        color = "red" if nes > 0 else "blue"
        ax.vlines(hits, ymax=ln[i], ymin=ln[i + 1], color=color, lw=0.5, alpha=0.3)

    fig.patch.set_facecolor("white")
    if not interactive_plot:
        plt.ion()
    return fig
