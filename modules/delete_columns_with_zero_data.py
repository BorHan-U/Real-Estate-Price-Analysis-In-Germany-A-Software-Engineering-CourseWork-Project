"""
This module provides functionality to delete columns from a DataFrame
where the number of zero values exceeds a given threshold.

Functions:
- delete_columns_with_zero_data: Deletes columns from a DataFrame where
  the number of zero values exceeds a given threshold.
- main: Parses command-line arguments and deletes columns with zero values
  exceeding the threshold in the specified input CSV file.
"""

import argparse
from typing import Union
import pandas as pd
import numpy as np


def delete_columns_with_zero_data(data: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Deletes columns from a DataFrame where the number of zero values exceeds a given threshold.

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
    TypeError
        If the input data is not a pandas DataFrame or threshold is not an integer.
    ValueError
        If the DataFrame is empty or the threshold is negative.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    
    if not isinstance(threshold, int):
        raise TypeError("Threshold must be an integer.")

    if data.empty:
        raise ValueError("The DataFrame is empty.")

    if threshold < 0:
        raise ValueError("Threshold must be a non-negative integer.")

    columns_to_drop = []

    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            zero_count = (data[column] == 0).sum() + data[column].isna().sum()
            if zero_count > threshold:
                columns_to_drop.append(column)
        else:
            print(f"Column '{column}' is not numeric and was skipped.")

    if columns_to_drop:
        data = data.drop(columns=columns_to_drop)
        print(f"Dropped columns: {', '.join(columns_to_drop)}")
    else:
        print("No columns were dropped.")

    # If all columns were dropped, return an empty DataFrame with the original index
    if len(data.columns) == 0:
        return pd.DataFrame(index=data.index)

    return data


def main() -> None:
    """
    Parses command-line arguments and deletes columns from a DataFrame
    where zero values exceed a given threshold.

    The filtered data is saved to a new CSV file.

    Raises
    ------
    SystemExit
        If the command-line arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description="Delete columns from a DataFrame where zero values exceed a given threshold."
    )
    parser.add_argument("file", type=str, help="Path to the input CSV file.")
    parser.add_argument(
        "threshold", type=int,
        help="Threshold for the maximum allowed number of zero values in a column."
    )
    parser.add_argument(
        "--output", type=str, default="filtered_data.csv",
        help="Path to save the filtered CSV file."
    )

    args = parser.parse_args()

    try:
        # Read the data from the CSV file
        data = pd.read_csv(args.file)
    except FileNotFoundError:
        print(f"Error: The file '{args.file}' was not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{args.file}' is empty.")
        return
    except Exception as e:
        print(f"Error reading the file: {str(e)}")
        return

    try:
        # Apply the column deletion based on zero values
        filtered_data = delete_columns_with_zero_data(data, args.threshold)

        # Save the filtered data to a CSV file
        filtered_data.to_csv(args.output, index=False)
        print(f"Filtered data saved to {args.output}")
    except (TypeError, ValueError) as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()
