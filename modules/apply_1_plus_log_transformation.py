"""
This module provides functionality to apply a log(1 + x) transformation
to specified columns in a pandas DataFrame.

Functions:
- apply_1_plus_log_transformation: Applies the log(1 + x) transformation
  to specified columns in a DataFrame.
- main: Parses command-line arguments and applies the transformation to
  the specified columns in the input CSV file.
"""

import argparse
import numpy as np
import pandas as pd


def apply_1_plus_log_transformation(data, columns_to_transform):
    """
    Applies the log(1 + x) transformation to specified columns in a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input data as a pandas DataFrame.
    columns_to_transform : list
        A list of column names to apply the transformation to.

    Returns
    -------
    pd.DataFrame
        The transformed data with log(1 + x) applied to specified columns.

    Raises
    ------
    ValueError
        If any column in columns_to_transform is not in the DataFrame.
    """
    transformed_data = data.copy()

    for column in columns_to_transform:
        if column not in transformed_data.columns:
            raise ValueError(f"Column '{column}' is not in the DataFrame.")
        try:
            # Handle non-numeric data types gracefully
            transformed_data[column] = pd.to_numeric(
                transformed_data[column], errors='coerce'
            )
            # Apply the log(1 + x) transformation
            transformed_data[column] = np.log1p(transformed_data[column])
        except Exception as exc:
            print(f"Error applying log(1 + x) transformation to column '{column}': {exc}")
            raise

    return transformed_data


def main():
    """
    Parses command-line arguments and applies the log(1 + x) transformation
    to the specified columns in the input CSV file.

    The transformed data is saved to a new CSV file.

    Raises
    ------
    SystemExit
        If the command-line arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description="Apply log(1 + x) transformation to specified columns in a DataFrame."
    )
    parser.add_argument("file", type=str, help="Path to the input CSV file.")
    parser.add_argument(
        "columns", nargs='+', type=str, help="Columns to apply the log(1 + x) transformation to."
    )
    parser.add_argument(
        "--output", type=str, default="transformed_data.csv",
        help="Path to save the transformed CSV file."
    )

    args = parser.parse_args()

    # Read the data from the CSV file
    data = pd.read_csv(args.file)

    # Apply the log(1 + x) transformation
    transformed_data = apply_1_plus_log_transformation(data, args.columns)

    # Save the transformed data to a CSV file
    transformed_data.to_csv(args.output, index=False)
    print(f"Transformed data saved to {args.output}")


if __name__ == "__main__":
    main()
