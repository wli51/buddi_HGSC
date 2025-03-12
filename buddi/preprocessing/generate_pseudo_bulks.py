from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData

import utils
from utils import CellDf

def generate_pseudo_bulk_from_props(
        in_adata: AnnData, 
        cell_df: CellDf, 
        props_df: pd.DataFrame, 
        num_cells: int, 
        cell_type_col: str, 
        cell_noise: List[np.array] = None, 
        use_sample_noise: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates pseudobulk expression profiles based on provided proportion data.

    :param in_adata: The AnnData object containing single-cell expression data.
    :param cell_df: Dictionary of cell type names mapped to subsetted AnnData objects.
    :param props_df: DataFrame containing cell type proportions for each sample.
    :param num_cells: Number of cells to sample for each pseudobulk profile.
    :param cell_type_col: Column name in in_adata.obs specifying the cell type labels.
    :param cell_noise: List of noise vectors for each cell type.
    :param use_sample_noise: Whether to apply additional noise to the pseudobulk profiles.
    :return: Tuple of (total proportion DataFrame, total expression DataFrame).
    """
    num_celltypes = props_df.shape[1]

    # Generate cell-specific noise if not provided
    if cell_noise is None:
        cell_noise = [np.random.lognormal(0, 0.1, in_adata.shape[1]) for _ in range(num_celltypes)]

    total_prop_list = []
    total_expr_list = []

    num_cells = num_cells if num_cells is not None else np.random.randint(200, 5000)
    count_df = utils.generate_count_from_props(props_df, num_cells)

    for samp_idx, count_profile in count_df.iterrows():
        if samp_idx % 100 == 0:
            print(f"Processing sample {samp_idx}")        

        # Initialize gene expression vector for pseudobulk
        sum_over_cells = np.zeros(in_adata.shape[1])

        for cell_idx, (cell_type, num_cell) in enumerate(count_profile.items()):

            ct_sum = utils.get_cell_type_sum(in_adata, cell_df[cell_type], num_cell)
            # Apply cell-specific noise
            ct_sum = np.multiply(ct_sum, cell_noise[cell_idx])
            sum_over_cells += ct_sum

        # Apply sample noise if enabled
        if use_sample_noise:
            sample_noise = np.random.lognormal(0, 1, in_adata.shape[1])
            sum_over_cells *= sample_noise
            sum_over_cells *= np.random.lognormal(0, 0.1, 1)[0]
            sum_over_cells *= np.random.lognormal(0, 0.1, in_adata.shape[1])
            sum_over_cells = np.random.poisson(sum_over_cells)[0]

        # Convert to DataFrame
        sum_over_cells_df = pd.DataFrame(sum_over_cells).T
        sum_over_cells_df.columns = in_adata.var['gene_ids']

        total_expr_list.append(sum_over_cells_df)
        total_prop_list.append(props_df.iloc[samp_idx].values)

    # Combine into final DataFrames
    total_prop_df = pd.concat(total_prop_list, axis=0)
    total_expr_df = pd.concat(total_expr_list, axis=0)

    return total_prop_df, total_expr_df