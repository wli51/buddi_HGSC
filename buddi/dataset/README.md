# dataset
The `dataset` folder contains the implementation of BuDDI compatible generating functions

## Files 
The `dataset` folder contains the following files
```
./dataset/
├── dataset.py
├── utils.py
└── README.md
```
- `dataset.py` contains the following functions:
    - `get_dataset`: custom buddi dataset generating function that can yield custom order input and output tuples meant for experimentation with buddi architecture
    - `_dataset_generator` and `_dataset_output_signature`: helper functions called by `get_dataset`
    - `dataset_generator`: buddi 4 supervised dataset generator
    - `get_output_signature`: buddi 4 supervised dataset output signature
    - `dataset_generator_unsupervised`: buddi 4 unsupervised dataset generator
    - `get_output_signature_unsupervised`: buddi 4 unsupervised dataset output signature

- `utils.py` currently only has 1 function `train_validation_split` to perform train validation split from generated dataset

## Usage
```python
import numpy as np

# data modality numpy arrays (generate dummy ones here)
num_samples = 1000
X_kp = np.random.rand(num_samples, 7000) # normalized RNA expression
label_kp = encoded['sample_id'].values[idx_sc_train,]

num_classes = 50 # one hot encoded sample id/label
label_kp = np.eye(num_classes)[
    np.random.randint(0, num_classes, size=num_samples)
]

num_classes = 2 # one hot encoded perturbation
drug_kp = np.eye(num_classes)[
    np.random.randint(0, num_classes, size=num_samples)
]

num_classes = 2 # one hot encoded technology
bulk_kp = np.eye(num_classes)[
    np.random.randint(0, num_classes, size=num_samples)
]

# cell type proportion
y_kp = np.random.rand(num_samples, 10)
y_kp = y_kp / y_kp.sum(axis=1, keepdims=True)

## Generate supervised dataset with
dataset_supervised = tf.data.Dataset.from_generator(
    lambda: dataset_generator(X_kp, y_kp, label_kp, drug_kp, bulk_kp),
    output_signature=get_output_signature(X_kp, y_kp, label_kp, drug_kp, bulk_kp)
)

## Similarly, unsupervised dataset can be generated with
# where X_unkp is expression, label_unkp, drug_unkp, bulk_unkp are one hot encoded metadata
dataset_unsupervised = tf.data.Dataset.from_generator(
    lambda: dataset_generator_unsupervised(X_unkp, label_unkp, drug_unkp, bulk_unkp),
    output_signature=get_output_signature_unsupervised(X_unkp, label_unkp, drug_unkp, bulk_unkp)
)
```