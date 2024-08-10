"""
This module provides functionality to count the number of zero values and
NaN values in each column of a pandas DataFrame.

Functions:
- count_null_data: Counts the number of zero values and NaN values in each
  column of a DataFrame.
- main: Parses command-line arguments and counts the zero and NaN values
  in the specified input CSV file.
"""

import argparse
import pandas as pd


def count_null_data(data):
    """
    Counts the number of zero values and NaN values in each column of the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input data as a pandas DataFrame.

    Prints
    ------
    - The count of zero values and NaN values for each column with non-zero count.
    - A message if there are no zero or NaN values.

    Raises
    ------
    TypeError
        If the input data is not a pandas DataFrame.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    missing_counts = (data == 0).sum()  # Count zero values in each column
    sorted_columns = missing_counts.sort_values(ascending=False)
    no_missing_data = True

    for column, count in sorted_columns.items():
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(data[column]):
            # Count NaN values in the column
            nan_count = data[column].isna().sum()
            count += nan_count
            if count > 0:
                print(
                    f"Column '{column}': {count} zero or NaN values "
                    f"(0 values: {missing_counts[column]}, NaN values: {nan_count})"
                )
                no_missing_data = False
        else:
            print(f"Column '{column}' is not numeric and was skipped.")

    if no_missing_data:
        print("There are no zero or NaN values in any numeric columns!")


def main():
    """
    Parses command-line arguments and counts the number of zero and NaN values
    in each column of the input CSV file.

    Raises
    ------
    SystemExit
        If the command-line arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description="Count the number of zero and NaN values in each column of a DataFrame."
    )
    parser.add_argument("file", type=str, help="Path to the input CSV file.")
    args = parser.parse_args()

    # Read the data from the CSV file
    data = pd.read_csv(args.file)

    # Call the count_null_data function
    count_null_data(data)


if __name__ == "__main__":
    main()
