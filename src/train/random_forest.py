import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib


def rf_train(x_train, y_train, logger, config):
    """
    Trains a Random Forest classifier using the provided training data.

    Parameters:
    -----------
    x_train : pd.DataFrame
        The feature matrix for the training data.
    y_train : pd.Series
        The target values for the training data.
    logger : logging.Logger
        The logger object for logging training progress and results.
    config : dict
        A dictionary containing configuration parameters for training the Random Forest model.
        It includes the following keys:
        - n_estimators : int
            The number of trees in the forest.
        - max_depth : int
            The maximum depth of the trees.
        - min_samples_split : int
            The minimum number of samples required to split an internal node.

    Returns:
    --------
    None
        The function saves the trained Random Forest model to the specified file path.
    """
    param_grid = {
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "min_samples_split": config["min_samples_split"],
    }
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=param_grid,
        cv=5,
        return_train_score=True,
    )
    start_time = time.time()
    logger.info("----------Random Forest training start----------")
    grid_search.fit(x_train, y_train)
    end_time = time.time()

    best_model = grid_search.best_estimator_
    cv_results = grid_search.cv_results_
    best_index = np.argmax(cv_results["mean_test_score"])
    train_accuracy = cv_results["mean_train_score"][best_index]
    val_accuracy = grid_search.best_score_
    print(f"train_accuracy = {train_accuracy}, val_accuracy = {val_accuracy}")
    logger.info(f"train_accuracy = {train_accuracy}, val_accuracy = {val_accuracy}")
    logger.info(f"Random Forest training time: {end_time - start_time}")
    logger.info("----------Random Forest training end----------")
    joblib.dump(best_model, config["rf_model_save_path"])
