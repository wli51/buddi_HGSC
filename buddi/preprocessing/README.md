# Buddi single cell RNA-seq preprocessing

## This preprocessing module currently contains 3 python source files
1. **`generate_pseudo_bulks.py`**:
Contains the main pseudo-bulk generationf functions `generate_pseudo_bulk_from_counts` and `generate_pseudo_bulk_from_props` that generates a pseudobulk expression profile from:
    - `props_df`/`counts_df`: a proportion or count matrix (pandas `Dataframe`)
    - `in_adata`: an cell-type annotated `anndata` object
    - `cell_df`: a `dict` mapping cell types to the corresponding subsetted `anndata` objects at the obervation level
    - `cell_noise`: an optional `list` of numpy `array` specifying the cell type specific expression noise
    - `use_sample_noise`: `bool` switch for application of random sample-level noise
    - `sample_noise_kwargs`: optional `kwargs` to modify behavior of sample-level noise application, pass on to `apply_sample_wise_noise` from utils
2. **`utils.py`**
Contains the:
- Utility functions:
    - `subset_adata_by_cell_type`: produces `cell_df`
    - `generate_counts_from_props`: converts proportion matrix to count matrix
    - `generate_prop_from_counts`: samples count matrix from props given cell count. 
- Functions to generate cell type mixture profiles in the form of proportion or count matrices:
    - `generate_log_normal_counts`: samples random log normal distributed relative abundance of cell type and normalizes to a proportion matrix
    - `generate_single_celltype_dominant_props`: generates single cell type dominant proportions
    - `get_true_proportions`: computes true proportion from annotated `anndata`
    - `generate_random_similar_props`: generates proportion matrices similar to a base proportion profile regulated by correlation. When used in conjunction with `get_true_proportions`, can produce somewhat realistic cell type proportions. 
- Helper functions called by the `generate_pseudo_bulks` functions:
    - `get_cell_type_sum`: sample and sum single cell expressions to produce single cell psuedo-bulk profiles
    - `apply_sample_wise_noise`: applies sample-wise noise
    
3. `sc_preprocess.py`
- Old function prior to refactoring, no longer used # TODO remote in some future version

## Workflows 

Pseudobulks can be generated from the following workflows:

0. **Pre-requisite** imports and generating `cell_df`
```python
from buddi.preprocessing import utils
from buddi.preprocessing import generate_pseudo_bulks

# processed, annotated adata
adata = sc.read_h5ad("PATH/TO/H5AD")

CELL_TYPE_COL = "[cell type column name]"

present_cell_types = subset_adata.obs[CELL_TYPE_COL].unique().to_list()

# subset anndata to cell type specific ones
cell_df = utils.subset_adata_by_cell_type(
    in_adata=adata, 
    cell_type_col=CELL_TYPE_COL,
    cell_order=cell_order
)
```

1. **Realistic Pseudobulk**:
```python
# Compute true proportion
true_props_df = utils.get_true_proportions(
    in_adata=in_adata,
    cell_type_col=CELL_TYPE_COL,
    cell_order=cell_order    
)

realistic_props_df = utils.generate_random_similar_props(
    num_samp=50,
    props_df=true_props_df
)

realistic_counts_df = utils.generate_counts_from_props(
    prop_df=realistic_props_df,
    num_cells=1000, # can also be randomly sampled per sample
)

realistic_pseudobulk_df = generate_pseudo_bulk_from_counts(
    in_adata=subset_adata_missing_ct,
    cell_df=cell_df,
    count_df=realistic_counts_df,
    use_sample_noise=False
)

# Final output of workflow
psuedobulk_prop, pseudobulk_expr = realistic_props_df, realistic_pseudobulk_df
```

2. **Random Pseudobulk**:

```python
random_count_df = utils.generate_log_normal_counts(
    cell_order=cell_order, 
    num_cells=5000, 
    num_samples=50,
    present_cell_types=present_cell_types
)

random_prop_df = utils.generate_prop_from_counts(
    random_count_df,
)

random_pseudobulk_df = generate_pseudo_bulk_from_counts(
    in_adata=subset_adata_missing_ct,
    cell_df=cell_df,
    count_df=random_count_df,
    use_sample_noise=False
)

# Final output of workflow
psuedobulk_prop, pseudobulk_expr = random_prop_df, random_pseudobulk_df
```

3. **Single cell type dominant Pseudobulk**:

```python
single_cell_props_df = utils.generate_single_celltype_dominant_props(
    num_samp=10, # number of samples per cell type that is present
    cell_order=cell_order,
    present_cell_types=present_cell_types
)

single_cell_counts_df = utils.generate_counts_from_props(
    single_cell_props_df,
    num_cells=1000
)

single_cell_pseudobulk_df = generate_pseudo_bulk_from_counts(
    in_adata=subset_adata_missing_ct,
    cell_df=cell_df,
    count_df=single_cell_counts_df,
    use_sample_noise=False
)

# Final output of workflow
psuedobulk_prop, pseudobulk_expr = single_cell_props_df, single_cell_pseudobulk_df
```