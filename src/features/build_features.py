import pandas as pd
import numpy as np


def drop_cols(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    useless_cols: list[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Drops columns with over a specified fraction of
    missing values and removes useless columns.

    Parameters:
    -----------
    train_df: pandas.DataFrame
        The DataFrame to drop columns from for training data.
    test_df: pandas.DataFrame
        The DataFrame to drop columns from for test data.
    useless_cols: list
        A list of column names to drop.
    threshold: float
        The maximum allowed fraction of missing values in a column.
        Columns with missing values exceeding this threshold will be dropped.
        Defaults to 0.5.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the modified training DataFrame
        and the modified test DataFrame,
        both without the dropped columns.
    """
    missing_ratio = train_df.isna().mean()
    cols_drop_by_missing = missing_ratio[missing_ratio > threshold].index
    cols_drop_all = cols_drop_by_missing.union(pd.Index(useless_cols))
    train_df = train_df.drop(cols_drop_all, axis=1)
    test_df = test_df.drop(cols_drop_all, axis=1)
    return train_df, test_df


def fill_missing_with_mean(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Fills missing values in columns with numerical values
    using the mean of the column from the training data.

    Parameters:
    -----------
    train_df: pandas.DataFrame
        The DataFrame to fill missing values from for training data.
    test_df: pandas.DataFrame
        The DataFrame to fill missing values from for test data.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the modified training DataFrame
        and the modified test DataFrame, both with missing values
        filled using the mean of the column from the training data.
    """
    numeric_cols = test_df.select_dtypes(include=[np.number], exclude=[bool]).columns
    train_df[numeric_cols] = train_df[numeric_cols].fillna(
        train_df[numeric_cols].mean()
    )
    test_df[numeric_cols] = test_df[numeric_cols].fillna(train_df[numeric_cols].mean())
    return train_df, test_df


def encode_category(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    binary_cols: list[str],
    nominal_cols: list[str],
) -> pd.DataFrame:
    """
    Encodes categorical features using one-hot encoding

    Parameters:
    -----------
    train_df: pandas.DataFrame
        The DataFrame to encode for training data.
    test_df: pandas.DataFrame
        The DataFrame to encode for test data.
    binary_cols: list
        A list of binary feature names to encode.
    nominal_cols: list
        A list of other categorical column names to encode.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the modified training DataFrame
        and the modified test DataFrame,
        both with the specified columns one-hot encoded.
    """
    for col in binary_cols:
        col_values = train_df[col].unique()
        col_map = {val: i for i, val in enumerate(col_values)}
        train_df[col] = train_df[col].replace(col_map)
        test_df[col] = test_df[col].replace(col_map)

    train_df = pd.get_dummies(data=train_df, columns=nominal_cols)
    test_df = pd.get_dummies(data=test_df, columns=nominal_cols)
    return train_df, test_df


def split_features_label(df: pd.DataFrame, label_col: str) -> tuple:
    """
    Splits a Pandas dataframe into features and label.

    Parameters:
    -----------
    df: pandas.DataFrame
       The Dataframe to be split.
    label_col: str
       The name of the column to be use as label.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        A tuple containing the DataFrame representing the features
        and the Series representing the label.
    """
    features = df.drop(label_col, axis=1)
    label = df[label_col]
    return features, label


def encode_name(train_df, test_df):
    """
    Encodes the "Name" column in train_df and test_df by extracting
    the title of the passengers and performing one-hot encoding.

    Parameters:
    -----------
    train_df : pd.DataFrame
        The dataframe containing the training data.
    test_df : pd.DataFrame
        The dataframe containing the test data.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the encoded train_df and test_df dataframes.
        The "Name" column is encoded as titles using one-hot encoding.
    """
    train_df["title"] = train_df["Name"].str.split("[,|\.]", expand=True)[1].str.strip()
    test_df["title"] = test_df["Name"].str.split("[,|\.]", expand=True)[1].str.strip()
    train_df = train_df.drop("Name", axis=1)
    test_df = test_df.drop("Name", axis=1)
    top3 = train_df["title"].value_counts()[:3].index.tolist()
    train_df.loc[~train_df["title"].isin(top3), "title"] = "other"
    test_df.loc[~test_df["title"].isin(top3), "title"] = "other"
    train_df = pd.get_dummies(data=train_df, columns=["title"])
    test_df = pd.get_dummies(data=test_df, columns=["title"])
    return train_df, test_df
