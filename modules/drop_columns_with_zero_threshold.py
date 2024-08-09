"""
This module provides functionality to drop columns from a DataFrame
where the number of zero values exceeds a given threshold.

Functions:
- drop_columns_with_zero_threshold: Drops columns from a DataFrame where
  the number of zero values exceeds the given threshold.
- main: Parses command-line arguments and drops columns with zero values
  exceeding the threshold in the specified input CSV file.
"""

import argparse
import pandas as pd


def drop_columns_with_zero_threshold(data, threshold):
    """
    Drops columns from the DataFrame where the number of zero values exceeds the given threshold.

    Parameters
    ----------
    data : pd.DataFrame
        The input data as a pandas DataFrame.
    threshold : int
        The maximum allowed number of zero values in a column before it is dropped.

    Returns
    -------
    pd.DataFrame
        The DataFrame with columns removed where zero values exceed the threshold.

    Raises
    ------
    ValueError
        If the DataFrame is empty or the threshold is negative.
    """
    if data.empty:
        raise ValueError("The DataFrame is empty.")

    if threshold < 0:
        raise ValueError("Threshold must be a non-negative integer.")

    # Ensure we only consider numeric columns
    numeric_data = data.select_dtypes(include=[int, float, bool])

    if numeric_data.empty:
        print("No numeric columns found. No columns were dropped.")
        return data

    zero_counts = (numeric_data == 0).sum()
    columns_to_drop = zero_counts[zero_counts > threshold].index

    if columns_to_drop.empty:
        print("No columns have zero values exceeding the threshold. No columns were dropped.")
    else:
        data = data.drop(columns=columns_to_drop)
        print(f"Dropped columns: {', '.join(columns_to_drop)}")

    return data


def main():
    """
    Parses command-line arguments and drops columns from a DataFrame
    where zero values exceed a given threshold.

    The filtered data is saved to a new CSV file.

    Raises
    ------
    SystemExit
        If the command-line arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description="Drop columns from a DataFrame where the number of zero values exceeds a given threshold."
    )
    parser.add_argument("file", type=str, help="Path to the input CSV file.")
    parser.add_argument(
        "threshold", type=int,
        help="Threshold for the maximum allowed number of zero values in a column."
    )
    parser.add_argument(
        "--output", type=str, default="filtered_data_for_zero_threshold.csv",
        help="Path to save the filtered CSV file."
    )

    args = parser.parse_args()

    # Read the data from the CSV file
    data = pd.read_csv(args.file)

    # Apply the column dropping based on zero values
    filtered_data = drop_columns_with_zero_threshold(data, args.threshold)

    # Save the filtered data to a CSV file
    filtered_data.to_csv(args.output, index=False)
    print(f"Filtered data saved to {args.output}")


if __name__ == "__main__":
    main()
