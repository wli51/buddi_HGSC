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

"""
Pseudo-bulk sample proportion generation utilities
"""

def generate_random_similar_props(
        num_samp: int, 
        base_prop: np.ndarray, ## TODO may be combine base_prop with cell_order?
        cell_order: List[str], 
        min_corr: float = 0.8
    ) -> pd.DataFrame:
    """
    Helper function that generates a proportion matrix where each sample's cell-type proportions correlated 
    to a given base proportion vector. 
    This is meant for generating pseudo-bulk samples similar to some ground truth dataset in terms of composition but with some variation.

    :param num_samp: Number of samples to generate.
    :param base_prop: Base proportion vector (1D array of shape (num_celltypes,)).
    :param cell_order: List of cell type names (column names for the output DataFrame).
    :param min_corr: Minimum correlation threshold with the base proportion.
    :return: DataFrame with shape (num_samp, num_celltypes), where each row is a sample's cell-type proportions.
    """
    total_prop_list = []

    ## Sample random proportion scaling factors until the scaled proportion 
    # is sufficiently correlated with the base proportion
    while len(total_prop_list) < num_samp:
        # Apply noise to base proportion vector
        noisy_prop = base_prop * np.random.lognormal(0, 1, len(base_prop))
        noisy_prop = noisy_prop / np.sum(noisy_prop)  # Normalize to sum to 1

        # Compute correlation coefficient with the base proportion
        corr_coef = np.corrcoef(noisy_prop, base_prop)[0, 1]

        if corr_coef >= min_corr:
            total_prop_list.append(noisy_prop)

    # Convert to DataFrame
    return pd.DataFrame(total_prop_list, columns=cell_order)