"""
This script preprocesses house pricing data
by cleaning and transforming it for further analysis.

The preprocessing steps include:
- Dropping unnecessary columns.
- Mapping quality ratings to numerical values.
- Counting and handling missing data.
- Separating categorical and numerical data.
- Generating histograms of numerical data before and after cleaning.
- Applying log transformations to selected numerical columns.

Usage:
    python preprocess_script.py <input_file> <output_file> <output_dir>

Arguments:
- input_file: Path to the input CSV file containing the raw data.
- output_file: Path where the cleaned data will be saved.
- output_dir: Directory where histogram plots will be saved.
"""

import argparse
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

# Add the root directory to the Python path
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))
# pylint: disable=wrong-import-position, import-error
from modules.count_null_data import count_null_data  # noqa: E402
from modules.delete_columns_with_zero_data import delete_columns_with_zero_data  # noqa: E402
from modules.separate_categorical_numerical import (  # noqa: E402
    separate_categorical_numerical
)
from modules.drop_columns_with_zero_threshold import (  # noqa: E402
    drop_columns_with_zero_threshold
)
from modules.apply_1_plus_log_transformation import (  # noqa: E402
    apply_1_plus_log_transformation
)
# pylint: enable=wrong-import-position, import-error


def plot_histograms(data, filename, output_dir):
    """
    Plot histograms for the given data and save the figure.
    Args:
        data (pd.DataFrame): Data to plot
        filename (str): Name of the output file
        output_dir (str): Directory to save the plot
    """
    n_cols = len(data.columns)
    n_rows = (n_cols + 3) // 4  # Round up to the nearest multiple of 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(data.columns):
        data[col].hist(ax=axes[i], bins=30)
        axes[i].set_title(col, fontsize=10)
        axes[i].tick_params(axis='both', which='major', labelsize=8)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Frequency', fontsize=8)
        axes[i].tick_params(axis='x', rotation=90)

    # Remove unused subplots
    for i in range(n_cols, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def preprocess_data(input_file, output_file, output_dir):
    """
    Preprocess the data by cleaning and transforming it for further analysis.
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the preprocessed CSV file.
        output_dir (str): Directory to save the plots.
    """
    data = pd.read_csv(input_file)

    # Preprocessing steps
    data = data.drop('Id', axis=1)

    mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    columns_to_map = ['GarageQual', 'GarageCond', 'PoolQC', 'FireplaceQu',
                      'KitchenQual', 'HeatingQC', 'BsmtCond', 'BsmtQual',
                      'ExterCond', 'ExterQual']

    for column in columns_to_map:
        data[column] = data[column].map(mapping)

    count_null_data(data)

    columns = data.columns.tolist()
    columns.insert(-1, 'Age')
    data['Age'] = data['YrSold'] - data['YearBuilt']
    columns.remove('YearBuilt')
    columns.remove('YrSold')
    data = data[columns]

    data = data.fillna(0)

    count_null_data(data)

    threshold = 900
    data = delete_columns_with_zero_data(data, threshold)

    count_null_data(data)
    numerical_cols = separate_categorical_numerical(data)

    numerical_data = data[numerical_cols].copy()

    # Plot histograms for initial numerical data
    plot_histograms(
        numerical_data,
        'numerical_data_histogram_plot.png',
        output_dir
    )

    column_to_delete = ['GarageQual', 'GarageCond', 'GarageYrBlt']
    numerical_data = numerical_data.drop(column_to_delete, axis=1)

    threshold_0 = 200
    numerical_data = drop_columns_with_zero_threshold(
        numerical_data,
        threshold_0
    )

    # Plot histograms after cleaning
    plot_histograms(
        numerical_data,
        'after_cleaning_numericalData_histogram_plot.png',
        output_dir
    )

    columns_to_transform = ['1stFlrSF', 'GrLivArea', 'LotArea', 'SalePrice']
    transformed_data = apply_1_plus_log_transformation(
        numerical_data,
        columns_to_transform
    )

    # Plot histograms for transformed data
    plot_histograms(
        transformed_data,
        'transformed_data_histogram_plot.png',
        output_dir
    )

    # Save the preprocessed data
    transformed_data.to_csv(output_file, index=False)


def main():
    """Main function to parse arguments and call preprocess_data."""
    parser = argparse.ArgumentParser(
        description="Preprocess house pricing data."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the preprocessed CSV file"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save the plots."
    )
    args = parser.parse_args()

    preprocess_data(args.input_file, args.output_file, args.output_dir)


if __name__ == "__main__":
    main()
