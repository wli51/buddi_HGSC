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
    :param use_sample_noise: Whether to apply additional sample noise.
    :param num_test_samples: Number of samples to reserve for the test set.
    :return: Tuple (total_prop_df, total_expr_df, test_prop_df, test_expr_df).
    """
    num_celltypes = len(cell_df)
    cell_order = list(cell_df.keys())

    # Generate cell-specific noise if not provided
    if cell_noise is None:
        cell_noise = [np.random.lognormal(0, 0.1, in_adata.shape[1]) for _ in range(num_celltypes)]

    total_prop_list, total_expr_list = [], []
    test_prop_list, test_expr_list = [], []

    for samp_idx in range(num_samples + num_test_samples):  # Last num_test_samples for test set
        
        # TODO: replace with formal logging or some progress bar
        if samp_idx % 100 == 0:
            print(f"Processing sample {samp_idx}")

        n_cells = num_cells if num_cells is not None else np.random.randint(200, 5000)

        # Generate proportion vector (true or lognormal)
        if use_true_prop:
            count_vec = utils.generate_true_counts(in_adata, n_cells, cell_type_col)
        else:
            count_vec = utils.generate_log_normal_counts(num_celltypes, n_cells)

        # Convert to DataFrame
        props_df = pd.DataFrame([count_vec], columns=cell_order)
        props_df = props_df.div(props_df.sum(axis=1), axis=0)

        # Initialize pseudobulk expression
        sum_over_cells = np.zeros(in_adata.shape[1])

        for cell_idx, cell_type_id in enumerate(cell_order):
            num_cell = count_vec[cell_idx]
            ct_sum = utils.get_cell_type_sum(in_adata, cell_df[cell_type_id], num_cell)
            ct_sum = np.multiply(ct_sum, cell_noise[cell_idx])

            sum_over_cells += ct_sum

        if use_sample_noise:
            sample_noise = np.random.lognormal(0, 1, in_adata.shape[1])
            sum_over_cells *= sample_noise
            sum_over_cells *= np.random.lognormal(0, 0.1, 1)[0]
            sum_over_cells *= np.random.lognormal(0, 0.1, in_adata.shape[1])
            sum_over_cells = np.random.poisson(sum_over_cells)[0]

        sum_over_cells_df = pd.DataFrame(sum_over_cells).T
        sum_over_cells_df.columns = in_adata.var['gene_ids']

        # Assign to training or test set
        if samp_idx < num_samples:
            total_prop_list.append(props_df)
            total_expr_list.append(sum_over_cells_df)
        else:
            test_prop_list.append(props_df)
            test_expr_list.append(sum_over_cells_df)

    prop_df = pd.concat(total_prop_list, axis=0)
    pseudo_bulk_df = pd.concat(total_expr_list, axis=0)

    if num_test_samples > 0:
        test_prop_df = pd.concat(test_prop_list, axis=0)
        test_expr_df = pd.concat(test_expr_list, axis=0)
    else:
        test_prop_df, test_expr_df = None, None

    return (
        prop_df, 
        pseudo_bulk_df, 
        test_prop_df, 
        test_expr_df
    )