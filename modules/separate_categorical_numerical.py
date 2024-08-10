"""
This module provides functionality to separate the columns of a DataFrame
into categorical and numerical columns.

Functions:
- separate_categorical_numerical: Separates the columns of a DataFrame into
  categorical and numerical columns.
- main: Parses command-line arguments and separates the columns.
"""

import argparse
import json

import pandas as pd


def separate_categorical_numerical(data):
    """
    Separates the columns of a DataFrame into categorical and numerical columns.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be processed.

    Returns
    -------
    tuple
        A tuple containing two lists - categorical columns and numerical columns.

    Raises
    ------
    ValueError
        If the DataFrame is empty or contains no columns.
    """
    if data.empty:
        raise ValueError("The DataFrame is empty. Cannot separate columns.")

    if len(data.columns) == 0:
        raise ValueError("The DataFrame contains no columns.")

    categorical_cols = []
    numerical_cols = []

    for column in data.columns:
        dtype = data[column].dtype

        if (isinstance(dtype, pd.CategoricalDtype) or 
            pd.api.types.is_object_dtype(dtype) or 
            pd.api.types.is_bool_dtype(dtype) or
            pd.api.types.is_datetime64_any_dtype(dtype)):
            categorical_cols.append(column)
        elif pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_complex_dtype(dtype):
            numerical_cols.append(column)
        else:
            print(
                f"Column '{column}' has an unrecognized data type: {dtype}. "
                "It will be ignored."
            )

    return categorical_cols, numerical_cols


def main():
    """
    Parses command-line arguments and separates the columns of a DataFrame
    into categorical and numerical columns.

    The results are saved to specified files or printed to the console.

    Raises
    ------
    SystemExit
        If the command-line arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Separate the columns of a DataFrame into categorical and "
            "numerical columns."
        )
    )
    parser.add_argument(
        "input_file", type=str,
        help="Path to the input CSV file containing the data."
    )
    parser.add_argument(
        "--output_categorical", type=str,
        help="Path to save the list of categorical columns as a JSON file."
    )
    parser.add_argument(
        "--output_numerical", type=str,
        help="Path to save the list of numerical columns as a JSON file."
    )

    args = parser.parse_args()

    try:
        # Load the data from the CSV file
        data = pd.read_csv(args.input_file)

        # Separate the columns into categorical and numerical
        categorical_cols, numerical_cols = separate_categorical_numerical(data)

        # Save the results to files if specified
        if args.output_categorical:
            with open(args.output_categorical, 'w', encoding='utf-8') as file:
                json.dump(categorical_cols, file)
            print(f"Categorical columns saved to {args.output_categorical}")

        if args.output_numerical:
            with open(args.output_numerical, 'w', encoding='utf-8') as file:
                json.dump(numerical_cols, file)
            print(f"Numerical columns saved to {args.output_numerical}")

        # Print the results
        print("Categorical Columns:", categorical_cols)
        print("Numerical Columns:", numerical_cols)

    except FileNotFoundError:
        print(f"Error: The file '{args.input_file}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{args.input_file}' is empty.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
