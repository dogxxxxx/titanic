import matplotlib.pyplot as plt


def plot_4_line_charts(train_loss, val_loss, train_accuracy, val_accuracy, save_path):
    """
    Plots four line charts for train loss, validation loss,
    train accuracy, and validation accuracy.

    Parameters:
    -----------
    train_loss : List[float]
        List of train loss values.
    val_loss : List[float]
        List of validation loss values.
    train_accuracy : List[float]
        List of train accuracy values.
    val_accuracy : List[float]
        List of validation accuracy values.
    save_path : str
        The path to save the plot.

    Returns:
    --------
    None
    """
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs[0, 0].plot(train_loss)
    axs[0, 1].plot(val_loss)
    axs[1, 0].plot(train_accuracy)
    axs[1, 1].plot(val_accuracy)

    axs[0, 0].set_title("train loss")
    axs[0, 1].set_title("val loss")
    axs[1, 0].set_title("train accuracy")
    axs[1, 1].set_title("val accuracy")
    axs[0, 0].set_ylabel("loss")
    axs[0, 1].set_ylabel("loss")
    axs[1, 0].set_ylabel("accuracy")
    axs[1, 1].set_ylabel("accuracy")
    axs[0, 0].set_xlabel("epoch")
    axs[0, 1].set_xlabel("epoch")
    axs[1, 0].set_xlabel("epoch")
    axs[1, 1].set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_2_line_charts(train_accuracy, val_accuracy, save_path):
    """
    Plots two line charts for train accuracy and validation accuracy.

    Parameters:
    -----------
    train_accuracy : List[float]
        List of train accuracy values.
    val_accuracy : List[float]
        List of validation accuracy values.
    save_path : str
        The path to save the plot.

    Returns:
    --------
    None
    """
    fig, axs = plt.subplots(1, 2, figsize=(8, 8))
    axs[0].plot(train_accuracy)
    axs[1].plot(val_accuracy)

    axs[0].set_title("train accuracy")
    axs[1].set_title("val accuracy")
    axs[0].set_ylabel("accuracy")
    axs[1].set_ylabel("accuracy")
    axs[0].set_xlabel("epoch")
    axs[1].set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
