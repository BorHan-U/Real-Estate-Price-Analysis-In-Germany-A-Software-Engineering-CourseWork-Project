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
    data : pd.DataFrame or any other type
        The input data, expected to be a pandas DataFrame.

    Returns
    -------
    dict
        A dictionary containing the counts of zero and NaN values for each column.

    Raises
    ------
    TypeError
        If the input data is not a pandas DataFrame.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    result = {}
    no_missing_data = True

    for column in data.columns:
        if (pd.api.types.is_numeric_dtype(data[column]) and
        not pd.api.types.is_bool_dtype(data[column])):
            zero_count = (data[column] == 0).sum()
            nan_count = data[column].isna().sum()
            total_count = zero_count + nan_count

            if total_count > 0:
                result[column] = {
                    'total': total_count,
                    'zero': zero_count,
                    'nan': nan_count
                }
                no_missing_data = False
        else:
            result[column] = 'skipped'

    if no_missing_data:
        print("There are no zero or NaN values in any numeric columns!")
    else:
        for column, counts in result.items():
            if counts != 'skipped':
                print(f"Column '{column}': {counts['total']} zero or NaN values "
                      f"(0 values: {counts['zero']}, NaN values: {counts['nan']})")
            else:
                print(f"Column '{column}' is not numeric and was skipped.")

    return result


def main():
    """
    Parses command-line arguments and counts the number of zero and NaN values
    in each column of the input CSV file.

    Raises
    ------
    SystemExit
        If the command-line arguments are invalid or if the file cannot be read.
    """
    parser = argparse.ArgumentParser(
        description="Count the number of zero and NaN values in each column of a DataFrame."
    )
    parser.add_argument("file", type=str, help="Path to the input CSV file.")
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

    # Call the count_null_data function
    count_null_data(data)


if __name__ == "__main__":
    main()
