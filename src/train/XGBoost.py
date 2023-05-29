import time

import xgboost as xgb
import pandas as pd
import joblib
import numpy as np
from skopt import BayesSearchCV


def xgb_train(x_train: pd.DataFrame, y_train, config: dict(), logger):
    """
    Trains an XGBoost classifier using the provided training data.

    Parameters:
    -----------
    x_train : pd.DataFrame
        The feature matrix for the training data.
    y_train : array-like
        The target values for the training data.
    config : dict
        A dictionary containing configuration parameters for training the XGBoost model.
        It includes the following keys:
        - objective : str
            The objective function for the XGBoost classifier.
        - xgb_params : dict
            The parameter grid for hyperparameter tuning.
        - xgb_model_save_path : str
            The file path to save the trained XGBoost model.
    logger : logging.Logger
        The logger object for logging training progress.

    Returns:
    --------
    None
        The function saves the trained XGBoost model to the specified file path.
    """
    model = xgb.XGBClassifier(objective=config["objective"])
    param_grid = config["xgb_params"]
    bayessearch = BayesSearchCV(
        model,
        param_grid,
        scoring="accuracy",
        cv=5,
        n_iter=100,
        refit=True,
        return_train_score=True,
    )

    start_time = time.time()
    logger.info("----------XGBoost training start----------")
    bayessearch.fit(x_train, y_train)
    end_time = time.time()

    best_model = bayessearch.best_estimator_
    cv_results = bayessearch.cv_results_
    best_index = np.argmax(cv_results["mean_test_score"])
    train_accuracy = cv_results["mean_train_score"][best_index]
    val_accuracy = bayessearch.best_score_
    print(f"train_accuracy = {train_accuracy}, val_accuracy = {val_accuracy}")
    logger.info(f"train_accuracy = {train_accuracy}, val_accuracy = {val_accuracy}")
    logger.info(f"XGBoost training time: {end_time - start_time}")
    logger.info("----------XGBoost training end----------")
    joblib.dump(best_model, config["xgb_model_save_path"])
