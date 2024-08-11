"""
This module provides functionality to plot a boxplot of specified columns
in a DataFrame and save the plot to a file.

Functions:
- plot_boxplot: Plot a boxplot of the specified columns
in a DataFrame and save the plot to a file.
- main: Parses command-line arguments and plots the boxplot.
"""

import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class PlotSaveError(Exception):
    """Custom exception for errors during plot saving."""


def plot_boxplot(df: pd.DataFrame, x_column: str,
                 y_column: str, output_dir: str) -> None:
    """
    Plot a boxplot of the specified columns in
    a DataFrame and save the plot to a file.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    x_column : str
        The column name to be used for the x-axis.
    y_column : str
        The column name to be used for the y-axis.
    output_dir : str
        The directory where the plot will be saved.

    Raises
    ------
    ValueError
        If the DataFrame is empty, the specified
                columns do not exist, or the y_column is not numeric.
    PlotSaveError
        For any error that occurs during file writing.
    """
    if df.empty:
        raise ValueError("The DataFrame is empty. Cannot plot boxplot.")

    if x_column not in df.columns or y_column not in df.columns:
        raise ValueError(
            f"Columns '{x_column}' or '{y_column}'"
            f"do not exist in the DataFrame."
        )

    if not pd.api.types.is_numeric_dtype(df[y_column]):
        raise ValueError(
            f"The y_column '{y_column}'"
            f"must be numeric to plot a boxplot."
        )

    data = df[[x_column, y_column]]

    print(f"Plotting Boxplot for {x_column} vs {y_column}")
    print(data.describe())

    fig, ax = plt.subplots(figsize=(14, 9))
    sns.boxplot(x=x_column, y=y_column, data=data, ax=ax)
    plt.xticks(rotation=90)
    plt.title(f'Boxplot of {y_column} by {x_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    name = f"Boxplot_of_{y_column}_by_{x_column}.png"

    try:
        os.makedirs(output_dir, exist_ok=True)
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, name))
        print(f"Boxplot saved as {os.path.join(output_dir, name)}")
    except OSError as exc:
        plt.close(fig)  # Close the figure in case of an error
        raise PlotSaveError(f"Error saving the boxplot: {exc}") from exc
    finally:
        plt.close(fig)  # Always close the figure


def main() -> None:
    """
    Parses command-line arguments and plots a boxplot of specified columns
    in a DataFrame.

    The plot is saved to the specified directory.

    Raises
    ------
    SystemExit
        If the command-line arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot a boxplot of specified columns in"
            "a DataFrame and save it to a file."
        )
    )
    parser.add_argument(
        "input_file", type=str,
        help="Path to the input CSV file containing the data."
    )
    parser.add_argument(
        "x_column", type=str,
        help="The column name to be used for the x-axis."
    )
    parser.add_argument(
        "y_column", type=str,
        help="The column name to be used for the y-axis."
    )
    parser.add_argument(
        "output_dir", type=str,
        help="The directory where the plot will be saved."
    )

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"Error: The file '{args.input_file}' was not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{args.input_file}' is empty.")
        return

    try:
        plot_boxplot(df, args.x_column, args.y_column, args.output_dir)
    except (ValueError, PlotSaveError) as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
