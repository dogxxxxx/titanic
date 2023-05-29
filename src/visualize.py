import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import utils


def bar_plot(df: pd.DataFrame, x: str, save_dir: str, y: str = "Survived") -> None:
    """
    Draw a bar plot using Seaborn's barplot function and save it.

    Parameters:
    -----------
    df(pandas.DataFrame): The dataframe to use for the plot.
    x(str): The column in the dataframe to use as x-axis.
            Must be categorical data with few categories.
    y(str): The column in the dataframe to use as y-axis.
            Defaults to 'Survived'.
    save_dir(str): The directory to save the figures.

    Returns:
    --------
    None
    """
    plot = sns.barplot(x=x, y=y, data=df)
    plt.title(f"{y} rate by {x}")
    plt.savefig(save_dir + f"{y} rate by {x}")
    plt.show()


def main():
    config = utils.load_config("config.yaml")
    df = pd.read_csv(config["train_path"])
    bar_plot(df=df, x="Sex", save_dir=config["plot_save_directory"])


if __name__ == "__main__":
    main()
