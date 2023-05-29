import joblib


def xgb_predict(test_df, model_path):
    """
    Performs predictions using a trained XGBoost model.

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
    prediction = model.predict(test_df)
    prediction = [int(x) for x in prediction]
    return prediction
