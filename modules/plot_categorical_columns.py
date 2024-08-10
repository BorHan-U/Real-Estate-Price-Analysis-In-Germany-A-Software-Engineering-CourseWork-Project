"""
This module provides functionality to plot bar charts for the value counts
of each categorical column in a DataFrame.

Functions:
- plot_categorical_columns: Plots bar charts for the value counts of each
  categorical column in a DataFrame.
- main: Parses command-line arguments and plots the categorical columns.
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class PlotSaveError(Exception):
    """Custom exception for errors during plot saving."""


def plot_categorical_columns(data, output_dir=None):
    """
    Plots bar charts for the value counts of each categorical column in the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data.
    output_dir : str, optional
        The directory where the plots will be saved. If None, plots are displayed.

    Raises
    ------
    ValueError
        If the DataFrame is empty or if all columns are non-categorical.
    PlotSaveError
        For any error that occurs during file writing.
    """
    if data.empty:
        raise ValueError("The DataFrame is empty. Cannot plot categorical columns.")

    # Filter out non-categorical columns and columns with too many unique values
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    categorical_columns = [
        col for col in categorical_columns if data[col].nunique() <= 20
    ]

    if len(categorical_columns) == 0:
        raise ValueError(
            "No categorical columns with a reasonable number of unique values to plot."
        )

    num_cols = len(categorical_columns)
    num_rows = (num_cols - 1) // 6 + 1
    fig, axes = plt.subplots(nrows=num_rows, ncols=6, figsize=(20, num_rows * 4))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    try:
        for i, column in enumerate(categorical_columns):
            value_counts = data[column].value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i])
            axes[i].set_title(f'Value Counts - {column}')
            axes[i].set_xlabel('Categories')
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)

        # Hide any unused axes
        for j in range(len(categorical_columns), len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plot_file = os.path.join(output_dir, "categorical_columns_plots.png")
            plt.savefig(plot_file)
            print(f"Plots saved to {plot_file}")
        else:
            plt.show()
    except Exception as exc:
        plt.close(fig)
        if output_dir:
            raise PlotSaveError(f"Error saving the plots: {exc}") from exc
        else:
            raise PlotSaveError(f"Error displaying the plots: {exc}") from exc
    finally:
        plt.close(fig)


def main():
    """
    Parses command-line arguments and plots bar charts for categorical columns
    in a DataFrame.

    The plots are saved to the specified directory or displayed.

    Raises
    ------
    SystemExit
        If the command-line arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot bar charts for categorical columns in a DataFrame and save "
            "them to a file."
        )
    )
    parser.add_argument(
        "input_file", type=str,
        help="Path to the input CSV file containing the data."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help=(
            "Directory where the plots will be saved. If not specified, "
            "plots will be shown."
        )
    )

    args = parser.parse_args()

    try:
        # Load the data from the CSV file
        data = pd.read_csv(args.input_file)

        # Plot the categorical columns
        plot_categorical_columns(data, args.output_dir)
    except FileNotFoundError:
        print(f"Error: The file '{args.input_file}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{args.input_file}' is empty.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except PlotSaveError as pse:
        print(f"Error: {pse}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
