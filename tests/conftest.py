import os
import numpy as np
import pandas as pd
import pytest

KEGG_PATH = os.path.join(os.path.dirname(__file__), "../blitzgsea/data/KEGG_2021_Human")


# ---------------------------------------------------------------------------
# 10-gene fixtures for unit tests with known exact values
# ---------------------------------------------------------------------------

@pytest.fixture
def abs_sig_10():
    """Absolute values of a 10-gene signature, descending: [10, 9, ..., 1]."""
    return np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])


@pytest.fixture
def sig_map_10():
    return {f"GENE_{i}": i for i in range(10)}


@pytest.fixture
def sig_df_10():
    """Signature DataFrame indexed by gene name, already sorted descending."""
    return pd.DataFrame(
        {"v": [float(v) for v in range(10, 0, -1)]},
        index=[f"GENE_{i}" for i in range(10)],
    )


# ---------------------------------------------------------------------------
# Medium fixtures for integration tests
# ---------------------------------------------------------------------------

@pytest.fixture
def medium_signature():
    """
    200-gene signature as a two-column DataFrame (gene, value), values descending.
    Genes GENE_0..GENE_19 are at the top (positive), GENE_180..GENE_199 at the bottom.
    """
    genes = [f"GENE_{i}" for i in range(200)]
    values = list(np.linspace(5.0, -5.0, 200))
    return pd.DataFrame({"gene": genes, "value": values})


@pytest.fixture
def synthetic_library():
    """
    Three gene sets with known expected enrichment direction.

    top_set  → genes at top of ranking  → positive ES
    bottom_set → genes at bottom        → negative ES
    random_set → scattered              → weak signal
    """
    return {
        "top_set": [f"GENE_{i}" for i in range(20)],
        "bottom_set": [f"GENE_{i}" for i in range(180, 200)],
        "random_set": [f"GENE_{i}" for i in range(0, 200, 13)],
    }


@pytest.fixture
def kegg_library():
    from blitzgsea.enrichr import read_gmt
    return read_gmt(KEGG_PATH)
