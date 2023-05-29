import logging

import yaml
import pandas as pd


def load_config(config_file_path: str) -> dict:
    """
    Loads configs from a YAML file and returns them as a dictionary.

    Parameters:
    config_file_path(str): The path to the YAML file containing the configs.

    Returns:
    dict: A dictionary containing the configs
    """
    with open(config_file_path) as file:
        config = yaml.unsafe_load(file)
    return config


def load_data(file_path):
    """
    Loads a dataset from a file into a pandas dataframe.

    Parameters:
    file_path(str): The path of the file to load.

    Returns:
    pandas.DataFrame: A dataframe containing the loaded dataset
    """
    df = pd.read_csv(file_path)
    return df


def set_logging(log_path):
    """
    Sets up logging configuration to log messages to a specified file.

    Parameters:
    -----------
    log_path : str
        The path to the log file where log messages will be written.

    Returns:
    --------
    logging.Logger
        The logger object associated with the logging configuration.
    """
    logging.basicConfig(
        filename=log_path,
        filemode="a",
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger()
    return logger


def submit(prediction, test_df: pd.DataFrame, submit_path: str):
    """
    Creates a submission file from model predictions.

    Parameters:
    -----------
    prediction : array-like
        The predicted values for the test data.
    test_df : pd.DataFrame
        The dataframe containing the test data.
    submit_path : str
        The path to save the submission file.

    Returns:
    --------
    None
    """
    submission = pd.DataFrame()
    submission["PassengerId"] = test_df["PassengerId"]
    submission["Survived"] = prediction
    submission.to_csv(submit_path, index=None)
