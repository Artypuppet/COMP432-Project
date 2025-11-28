"""
This module provides functions to load training and test data from various
data directories (raw, interim, processed) with automatic project root detection.
"""

import os
import pandas as pd
import numpy as np


def get_data_path(filename, data_dir="raw"):
    """
    Get the full path to a data file.
    
    This function automatically detects the project root by looking for
    README.md file, then constructs the path to the data file.
    
    Args:
        filename: Name of the CSV file (e.g., "train.csv")
        data_dir: Directory within data/ folder ("raw", "interim", or "processed")
    
    Returns:
        Full path to the data file
    """
    # Start from current working directory
    current = os.path.abspath(os.getcwd())
    project_root = current
    
    # Walk up the directory tree to find project root (marked by README.md)
    # The while condition checks until we go to the root directory i.e. /
    while project_root != os.path.dirname(project_root):
        readme_path = os.path.join(project_root, "README.md")
        if os.path.exists(readme_path):
            break
        project_root = os.path.dirname(project_root)
    
    # Construct the full path to the data file
    data_path = os.path.join(project_root, "data", data_dir, filename)
    
    return data_path


def load_train_data(data_path=None, data_dir="raw", filename=None, exclude_id=True):
    """
    Load training data from a CSV file.
    
    This function loads training data with labels, automatically handling
    the id column if present, and returns clean NumPy arrays.
    
    Args:
        data_path: Full path to CSV file (overrides data_dir and filename if provided)
        data_dir: Directory within data/ folder ("raw", "interim", or "processed")
        filename: Name of the CSV file (default: "train.csv")
        exclude_id: If True, exclude the id column from features (default: True)
    
    Returns:
        X: NumPy array of shape (n_samples, n_features) containing features
        y: NumPy array of shape (n_samples,) containing class labels
    """
    # Determine the full path to the data file
    if data_path is None:
        if filename is None:
            filename = "train.csv"
        data_path = get_data_path(filename, data_dir)
    
    # Read the CSV file using pandas
    df = pd.read_csv(data_path)
    
    # Check if the first column is an id column
    # We check both by column name and by position
    has_id = False
    if len(df.columns) > 0:
        first_col_name = str(df.columns[0]).lower()
        if first_col_name == 'id' or first_col_name.startswith('id'):
            has_id = True
    
    # Extract features (X) and labels (y)
    # The label is always the last column
    if exclude_id and has_id:
        # Skip the id column (first column) and the label column (last column)
        X = df.iloc[:, 1:-1].values
    else:
        # Include id column but exclude label
        X = df.iloc[:, :-1].values
    
    # Extract labels from the last column
    y = df.iloc[:, -1].values.astype(np.int64)
    
    return X, y


def load_test_data(data_path=None, data_dir="raw", filename=None, exclude_id=True, return_ids=False):
    """
    Load test data from a CSV file.
    
    This function loads test data without labels, automatically handling
    the id column if present, and returns a clean NumPy array.
    
    Args:
        data_path: Full path to CSV file (overrides data_dir and filename if provided)
        data_dir: Directory within data/ folder ("raw", "interim", or "processed")
        filename: Name of the CSV file (default: "test.csv")
        exclude_id: If True, exclude the id column from features (default: True)
        return_ids: If True, also return the ID column as a separate array (default: False)
    
    Returns:
        X: NumPy array of shape (n_samples, n_features) containing features
        ids: (optional) NumPy array of IDs if return_ids=True
    """
    # Determine the full path to the data file
    if data_path is None:
        if filename is None:
            filename = "test.csv"
        data_path = get_data_path(filename, data_dir)
    
    # Read the CSV file using pandas
    df = pd.read_csv(data_path)
    
    # Check if the first column is an id column
    has_id = False
    if len(df.columns) > 0:
        first_col_name = str(df.columns[0]).lower()
        if first_col_name == 'id' or first_col_name.startswith('id'):
            has_id = True
    
    # Extract IDs if requested
    ids = None
    if return_ids and has_id:
        ids = df.iloc[:, 0].values
    
    # Extract features (X)
    # Test data has no label column, so we just need to handle the id column
    if exclude_id and has_id:
        # Skip the id column (first column)
        X = df.iloc[:, 1:].values
    else:
        # Include all columns (or no id column present)
        X = df.iloc[:, :].values
    
    if return_ids:
        return X, ids
    return X

