import features


def rf_preprocess(train_df, test_df, config):
    """
    Preprocesses the train and test dataframes for Random Forest model.

    Parameters:
    -----------
    train_df : pd.DataFrame
        The raw input train dataframe to be preprocessed.
    test_df : pd.DataFrame
        The raw input test dataframe to be preprocessed.
    config : dict
        A dictionary containing configuration parameters for preprocessing.
        It includes the following keys:
        - useless_cols : list
            A list of columns to be dropped.
        - binary_cols : list
            A list of columns with binary values.
        - nominal_cols : list
            A list of nominal columns.
        - label_col : str
            The name of the column to be used as the label.

    Returns:
    --------
    Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray], pd.DataFrame]
        A tuple containing the preprocessed data:
        - x_train : Union[pd.DataFrame, np.ndarray]
            The feature matrix for the training data.
        - y_train : Union[pd.Series, np.ndarray]
            The target values for the training data.
        - test_df : pd.DataFrame
            The preprocessed test dataframe.
    """
    train_df, test_df = features.drop_cols(
        train_df=train_df, test_df=test_df, useless_cols=config["useless_cols"]
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
    x_train, y_train = features.split_features_label(train_df, config["label_col"])
    return x_train, y_train, test_df
