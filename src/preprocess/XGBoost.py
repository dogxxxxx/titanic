import features


def xgboost_preprocess(train_df, test_df, config):
    """
    Preprocesses the training and test data for XGBoost model.

    Parameters:
    -----------
    train_df : pandas.DataFrame
        The raw input training dataframe to be preprocessed.
    test_df : pandas.DataFrame
        The raw input test dataframe to be preprocessed.
    config : dict
        A dictionary containing configuration parameters for preprocessing the data.
        It includes the following keys:
        - useless_cols : list
            A list of columns to be dropped.
        - binary_cols : list
            A list of columns with binary values to be encoded.
        - nominal_cols : list
            A list of nominal columns to be one-hot encoded.
        - label_col : str
            The name of the column to be used as the target label.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame]
        A tuple containing the preprocessed data:
        - x_train : pandas.DataFrame
            The feature matrix for the training data.
        - y_train : pandas.Series
            The target values for the training data.
        - test_df : pandas.DataFrame
            The preprocessed test dataframe.
    """
    train_df, test_df = features.drop_cols(train_df, test_df, config["useless_cols"])
    train_df, test_df = features.encode_category(
        train_df=train_df,
        test_df=test_df,
        binary_cols=config["binary_cols"],
        nominal_cols=config["nominal_cols"],
    )
    train_df, test_df = features.encode_name(train_df=train_df, test_df=test_df)
    x_train, y_train = features.split_features_label(train_df, config["label_col"])
    return x_train, y_train, test_df
