from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from anndata import AnnData

CellDf = Dict[str, AnnData]

"""
scRNA seq data processing utilities
"""

def get_true_proportions(
        in_adata: AnnData, 
        cell_type_col: str
    ) -> pd.DataFrame:
    """
    Helper function that calculates the true proportion of cell types in the given AnnData object.
    Requires a user specificed column name in in_adata.obs that contains cell type labels.

    :param in_adata: The AnnData object containing single-cell expression data.
    :param cell_type_col: Column name in in_adata.obs specifying cell type labels.
    :return: DataFrame containing a single-row vector of cell type proportions.
    """
    prop_counts = in_adata.obs[cell_type_col].value_counts(normalize=True)
    return pd.DataFrame(prop_counts).T  # Return as a single-row DataFrame

def subset_adata_by_cell_type(
        in_adata: AnnData, 
        cell_type_col: str
    ) -> CellDf:
    """
    Constructs a dictionary mapping each cell type to a subset of the AnnData object
    containing only cells of that type.
    Requires a user specified column name in in_adata.obs that contains cell type labels.

    :param in_adata: The AnnData object containing single-cell expression data.
    :param cell_type_col: Column name in in_adata.obs specifying the cell type labels.
    :return: Dictionary where keys are cell type names and values are subsetted AnnData objects.
    When a cell type is not present in the input data, the corresponding value will be an empty AnnData object.
    """
    return {ctype: in_adata[in_adata.obs[cell_type_col] == ctype] for ctype in in_adata.obs[cell_type_col].unique()}