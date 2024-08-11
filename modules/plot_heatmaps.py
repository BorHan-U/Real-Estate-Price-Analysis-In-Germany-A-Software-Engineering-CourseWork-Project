"""
This module provides functionality to plot correlation matrix heatmaps
from a DataFrame and save them to a file.

Functions:
- plot_heatmaps: Plot correlation matrix heatmaps
from a DataFrame and save them to a file.
- main: Parses command-line arguments and plots the heatmaps.
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class PlotSaveError(Exception):
    """Custom exception for errors during plot saving."""


def plot_heatmaps(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot correlation matrix heatmaps from a DataFrame and save them to a file.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    output_dir : str
        The directory where the heatmaps will be saved.

    Raises
    ------
    ValueError
        If the DataFrame is empty or does not contain numeric columns.
    PlotSaveError
        If there's an error saving the plot file.
    """
    if df.empty:
        raise ValueError("The DataFrame is empty. Cannot plot heatmaps.")

    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("The DataFrame does not contain numeric columns.")

    corrmat = numeric_df.corr()

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # Plot the full correlation matrix heatmap
    sns.heatmap(corrmat, vmax=0.8, square=True, cmap="RdBu", ax=ax[0])
    ax[0].set_title('Correlation Matrix Heatmap')

    k = 10
    if 'SalePrice' in corrmat.columns:
        cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
        cm = np.corrcoef(df[cols].values.T)
        sns.set(font_scale=1.25)
        sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                    annot_kws={'size': 10}, yticklabels=cols.values,
                    xticklabels=cols.values, cmap="RdBu", ax=ax[1])
        ax[1].set_title('Top 10 most correlated variables with sale price')
    else:
        ax[1].set_visible(False)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the heatmap as a PNG file
    fig.tight_layout()
    try:
        fig.savefig(os.path.join(output_dir, "Correlation_Matrix_Heatmap.png"))
    except Exception as e:
        raise PlotSaveError(f"Error saving the heatmap: {e}") from e
    finally:
        plt.close(fig)


def main() -> None:
    """
    Parses command-line arguments and plots correlation matrix heatmaps
    from a DataFrame.

    The heatmaps are saved to the specified directory.

    Raises
    ------
    SystemExit
        If the command-line arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot correlation matrix heatmaps from a DataFrame and save "
            "them to a file."
        )
    )
    parser.add_argument(
        "input_file", type=str,
        help="Path to the input CSV file containing the data."
    )
    parser.add_argument(
        "output_dir", type=str,
        help="Directory where the heatmaps will be saved."
    )

    args = parser.parse_args()

    try:
        # Load the data from the CSV file
        df = pd.read_csv(args.input_file)

        # Plot the heatmaps
        plot_heatmaps(df, args.output_dir)
        print(f"Heatmaps saved to {args.output_dir}")
    except FileNotFoundError:
        print(f"Error: The file '{args.input_file}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{args.input_file}' is empty.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except PlotSaveError as pse:
        print(f"Error: {pse}")


if __name__ == "__main__":
    main()
