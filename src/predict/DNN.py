import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import models


def DNN_predict(test_tensor: torch.Tensor, config: dict) -> list:
    """
    Performs predictions using a trained DNN model.

    Parameters:
    -----------
    test_tensor : torch.Tensor
        The input tensor containing the test data.
    config : dict
        A dictionary containing configuration parameters for prediction.
        It includes the following keys:
        - hidden_dim : list of int
            The dimensions of the hidden layers in the DNN model.
        - model_save_path : str
            The path to the saved DNN model.

    Returns:
    --------
    prediction : list
        The predicted values based on the input test data.
    """
    input_dim = test_tensor.shape[1]
    nums_data = test_tensor.shape[0]
    hidden_dims = config["hidden_dim"]
    model_path = config["DNN_model_save_path"]

    test_dataset = TensorDataset(test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=nums_data, shuffle=False)
    model = models.DNN(input_dim=input_dim, hidden_dims=hidden_dims)
    model.load_state_dict(torch.load(model_path))

    prediction = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0]
            output = model(inputs).squeeze(1)
            prediction.extend(np.round(output.tolist()))
    prediction = [int(x) for x in prediction]
    return prediction
