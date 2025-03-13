from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData

import utils
from utils import CellDf

def generate_pseudo_bulk_from_counts(
        in_adata: AnnData, 
        cell_df: CellDf, 
        count_df: pd.DataFrame,
        cell_noise: List[np.array] = None, 
        use_sample_noise: bool = True,
        sample_noise_kwargs: Dict = {}
    ) -> pd.DataFrame:
    """
    Generates pseudobulk expression profiles based on provided count data.

    :param in_adata: The AnnData object containing single-cell expression data.
    :param cell_df: Dictionary of cell type names mapped to subsetted AnnData objects.
    :param count_df: DataFrame containing the number of cells to sample for each cell type.
    :param cell_noise: List of noise vectors for each cell type.
    :param use_sample_noise: Whether to apply additional noise to the pseudobulk profiles.
    :return: Tuple of (total proportion DataFrame, total expression DataFrame).
    """
    num_celltypes = count_df.shape[1]

    # Generate cell-specific noise if not provided
    if cell_noise is None:
        cell_noise = [np.random.lognormal(0, 0.1, in_adata.shape[1]) for _ in range(num_celltypes)]

    total_expr_list = []

    for samp_idx, (_, count_profile) in enumerate(count_df.iterrows()):
        if samp_idx % 100 == 0:
            print(f"Processing sample {samp_idx}")        

        # Initialize gene expression vector for pseudobulk
        sum_over_cells = np.zeros((1, in_adata.shape[1]))

        for cell_idx, (cell_type, cell_count) in enumerate(count_profile.items()):

            ct_sum = utils.get_cell_type_sum(in_adata, cell_df[cell_type], cell_count)
            # Apply cell-specific noise
            ct_sum = np.multiply(ct_sum, cell_noise[cell_idx])
            sum_over_cells += ct_sum

        # Apply sample noise if enabled
        if use_sample_noise:
            sum_over_cells = utils.apply_sample_wise_noise(
                sum_over_cells, **sample_noise_kwargs)

        # Convert to DataFrame
        sum_over_cells_df = pd.DataFrame(sum_over_cells)
        sum_over_cells_df.columns = in_adata.var['gene_ids']

        total_expr_list.append(sum_over_cells_df)

    # Combine into final DataFrames
    total_expr_df = pd.concat(total_expr_list, axis=0)

    return total_expr_df

def generate_random_pseudo_bulk_profiles(
        in_adata: AnnData, 
        num_samples: int, 
        num_cells: int, 
        use_true_prop: bool, 
        cell_df: CellDf, 
        cell_type_col: str, 
        cell_noise: List[np.ndarray] = None, 
        use_sample_noise: bool = True,
        num_test_samples: int = 100
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates pseudobulk expression matrices by generatiing cell type proportions completely at random 
    or similar to some reference proportion profile and then sampling cells from each cell type accordingly.
    Also optionally generates a test set of samples.

    :param in_adata: The AnnData object containing single-cell RNA-seq data.
    :param num_samples: Number of pseudobulk samples to generate.
    :param num_cells: Number of cells to simulate per sample.
    :param use_true_prop: Whether to use true cell type proportions.
    :param cell_df: Dictionary mapping cell types to subsetted AnnData objects.
    :param cell_type_col: Column name in in_adata.obs specifying cell type labels.
    :param cell_noise: List of noise vectors for each cell type.
    :param use_sample_noise: Whether to apply additional noise to the pseudobulk profiles.
    :return: Tuple of (total proportion DataFrame, total expression DataFrame).
    """
    
    count_df = utils.generate_count_from_props(props_df, num_cells)

    return generate_pseudo_bulk_from_counts(
        in_adata=in_adata, 
        cell_df=cell_df, 
        count_df=count_df,
        cell_noise=cell_noise, 
        use_sample_noise=use_sample_noise,
        sample_noise_kwargs=sample_noise_kwargs
    )