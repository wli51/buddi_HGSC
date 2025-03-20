# models
The *models* folder contains the implementation of BuDDI models

## BuDDI Models
Currently, there are two BuDDI models, `buddi3` and `buddi4`.

1. `buddi3` is the base BuDDI VAE that learns and disentangles variations from **sample**, **sequencing technology**, and **cell type composition** and has one additional **slack** latent space that would absorb any other significant variations not captured by aforementioned latent spaces.

2. `buddi4` additionally incorporates the variation from a **perturbation** (drug, disease, etc.) on top of what `buddi3` is already learning.


## Files 
The *models* folder contains the following files
```
./models/
├── __init__.py
├── buddi3.py
├── buddi4.py
├── components.py
├── layers.py
├── losses.py
└── README.md
```

- `buddi3.py`(TODO) and `buddi4.py` contains the `buddi3` and `buddi4` model building function and training function. 
- `components.py` contains the builder function for encoder, decoder and classifier branches that is utilized by the model classes.
- `layers.py` currently only contains a single tensorflow Layer class that samples from the VAE latent space parameters.
- `losses.py` contains the loss function generators that returns functions with signature `Fn(y_true, y_pred)` that can directly be used in tensorflow `model.compile(.., loss=[Fn])`