import joblib
import pandas as pd


def rf_predict(test: pd.DataFrame, model_path: str) -> list:
    """
    Performs predictions using a trained Random Forest model.

    Parameters:
    -----------
    test : pandas.DataFrame
        The input test data to be predicted.
    model_path : str
        The path to the saved Random Forest model.

    Returns:
    --------
    prediction : array-like
        The predicted values based on the input test data.
    """
    model = joblib.load(model_path)
    prediction = model.predict(test)
    prediction = [int(x) for x in prediction]
    return prediction
