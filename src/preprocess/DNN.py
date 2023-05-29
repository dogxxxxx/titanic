from sklearn.model_selection import train_test_split
import torch
import pandas as pd

import features


def _dataframe_to_tensor(
    train_x: pd.DataFrame, test_x: pd.DataFrame, train_y: pd.Series
) -> tuple:
    """
    Transforms the features and labels
    from pandas DataFrames to PyTorch tensors.

    Parameters:
    -----------
    train_x : pd.DataFrame
        The DataFrame containing the training features.
    test_x : pd.DataFrame
        The DataFrame containing the testing features.
    train_y : pd.Series
        The Series containing the training labels.

    Returns:
    --------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing the PyTorch tensors representing
        the training features, training labels,
        and testing features respectively.
    """
    bool_cols = train_x.select_dtypes(include=bool).columns
    train_x[bool_cols] = train_x[bool_cols].astype("int64")
    test_x[bool_cols] = test_x[bool_cols].astype("int64")
    train_x_tensor = torch.tensor(train_x.values, dtype=torch.float32)
    test_x_tensor = torch.tensor(test_x.values, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y.values, dtype=torch.float32)
    return train_x_tensor, train_y_tensor, test_x_tensor


def DNN_preprocess(train_df, test_df, config):
    """
    Processes raw data into training, validation, and testing datasets for a DNN model.

    Parameters:
    -----------
    train_df : pd.DataFrame
        The raw input train dataframe to be preprocessed.
    test_df : pd.DataFrame
        The raw input test dataframe to be preprocessed.
    config : dict
        The dictionary with the following keys:
        - useless_cols : list
            A list of columns to be dropped.
        - binary_cols : list
            A list of columns with boolean values.
        - nominal_cols : list
            A list of nominal columns.
        - test_ratio : float
            The proportion of data to be used for validation.
            Should be a value between 0 and 1.

    Returns:
    --------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing the following tensors:
        - x_train : torch.Tensor
            The feature matrix for the training data.
        - y_train : torch.Tensor
            The target values for the training data.
        - x_val : torch.Tensor
            The feature matrix for the validation data.
        - y_val : torch.Tensor
            The target values for the validation data.
        - x_test : torch.Tensor
            The feature matrix for the testing data.
    """
    train_df, test_df = features.drop_cols(
        train_df=train_df,
        test_df=test_df,
        useless_cols=config["useless_cols"],
        threshold=0.5,
    )
    train_df, test_df = features.fill_missing_with_mean(
        train_df=train_df, test_df=test_df
    )
    train_df, test_df = features.encode_category(
        train_df=train_df,
        test_df=test_df,
        binary_cols=config["binary_cols"],
        nominal_cols=config["nominal_cols"],
    )
    train_df, test_df = features.encode_name(train_df=train_df, test_df=test_df)
    train_x, train_y = features.split_features_label(train_df, config["label_col"])
    train_x, train_y, x_test = _dataframe_to_tensor(
        train_x=train_x, test_x=test_df, train_y=train_y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        train_x, train_y, test_size=config["test_ratio"]
    )
    return x_train, y_train, x_val, y_val, x_test
