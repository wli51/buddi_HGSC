import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(dataset, test_size=0.2, shuffle=True, random_state=None):
    """
    Splits a TensorFlow dataset into train and validation sets.

    Parameters:
    - dataset: `tf.data.Dataset` object
    - test_size: Fraction of data to be used for validation
    - shuffle: Whether to shuffle before splitting
    - random_state: Random seed for reproducibility

    Returns:
    - train_dataset: `tf.data.Dataset` object
    - val_dataset: `tf.data.Dataset` object
    """
    # Convert dataset to NumPy arrays
    data_list = list(dataset.as_numpy_iterator())  # Extract as a list of tuples

    # Unpack inputs and labels (if applicable)
    if isinstance(data_list[0], tuple):  # Supervised dataset with (X, y)
        X, y = zip(*data_list)
        X, y = np.array(X), np.array(y)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle, random_state=random_state
        )

        # Convert back to TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    else:  # Unsupervised dataset (only X)
        X = np.array(data_list)
        X_train, X_val = train_test_split(
            X, test_size=test_size, shuffle=shuffle, random_state=random_state
        )

        # Convert back to TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(X_train)
        val_dataset = tf.data.Dataset.from_tensor_slices(X_val)

    return train_dataset, val_dataset