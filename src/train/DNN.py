import time

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import utils
import models


def DNN_train(x_train, y_train, x_val, y_val, logger, config):
    """
    Trains a DNN model with provided training and validation data

    Parameters:
    -----------
    x_train: torch.tensor
        The feature matrix for training data.
    y_train: torch.tensor
        The target values for training data.
    x_val: torch.tensor
        The feature matrix for validation data.
    y_val: torch.tensor
        The target values for validation data.
    logger: logging.RootLogger
        The logger to document training process.
    config: dict
        A dictionary containing configuration parameters
        for training the DNN model.
        It includes the following keys:
        - hidden_dim: List[int]
            The dimensions of the hidden layers in DNN model.
        - epoch: int
            The number of training epochs
        - learning_rate: float
            The learning rate for the optimizer.
        - batch_size: int
            The batch size used during training
        - patience: int
            The number of epochs to wait for improvement
            in validation loss before early stopping.
        - model_save_path: str
            The directory path to save the best model.
        - plot_save_path: str
            The directory path to save plots.

    Returns:
    --------
    None
    """
    num_feature = x_train.shape[1]
    hidden_dims = config["hidden_dim"]
    num_epoch = config["epoch"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    num_train_sample = x_train.shape[0]
    num_val_sample = x_val.shape[0]
    patience = config["patience"]
    model_save_path = config["DNN_model_save_path"]

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = models.DNN(input_dim=num_feature, hidden_dims=hidden_dims)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_loss_history = []
    val_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []
    best_val_loss = np.inf
    patience_count = 0
    start_time = time.time()
    logger.info("----------DNN training start----------")
    for epoch in range(num_epoch):
        train_loss = 0.0
        train_correct = 0
        model.train()
        for feature, label in train_loader:
            # Forward pass
            outputs = model(feature)
            loss = criterion(outputs, label.unsqueeze(1))
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Compute accuracy
            train_prediction = (outputs >= 0.5).squeeze()
            train_correct += (train_prediction == label.squeeze()).sum().item()
            train_loss += loss.item()
        train_accuracy = train_correct / num_train_sample
        train_loss /= len(train_loader)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            val_correct = 0
            for feature, label in val_loader:
                outputs = model(feature)
                loss = criterion(outputs, label.unsqueeze(1))
                val_prediction = (outputs >= 0.5).squeeze()
                val_correct += (val_prediction == label.squeeze()).sum().item()
                val_loss += loss.item()
            val_accuracy = val_correct / num_val_sample
            val_loss /= len(val_loader)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_accuracy_history.append(train_accuracy)
        val_accuracy_history.append(val_accuracy)
        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                f"train_accuracy={train_accuracy:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_accuracy={val_accuracy:.4f}"
            )
            logger.info(
                f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                f"train_accuracy={train_accuracy:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_accuracy={val_accuracy:.4f}"
            )
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_model = model.state_dict()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(
                    f"Early stopping after {epoch+1} epochs, "
                    f"best_val_loss={best_val_loss}, "
                    f"val_accuracy={best_val_accuracy}"
                )
                logger.info(
                    f"Early stopping after {epoch+1} epochs, "
                    f"best_val_loss={best_val_loss}, "
                    f"val_accuracy={best_val_accuracy}"
                )
                torch.save(best_model, model_save_path)
                break
    end_time = time.time()
    logger.info(f"DNN training time: {end_time - start_time}")
    logger.info("----------DNN training end----------")

    utils.plot_4_line_charts(
        train_loss=train_loss_history,
        val_loss=val_loss_history,
        train_accuracy=train_accuracy_history,
        val_accuracy=val_accuracy_history,
        save_path=config["plot_save_directory"] + "DNN_loss",
    )
