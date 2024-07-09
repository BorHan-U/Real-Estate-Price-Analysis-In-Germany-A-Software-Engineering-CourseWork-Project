import numpy as np
import pandas as pd

from modules.modules import (
    apply_1_plus_log_transformation,
    count_null_data,
    delete_columns_with_zero_data,
    drop_columns_with_zero_threshold,
    separate_categorical_numerical,
)

# Tests for count_null_data function


def test_missing_values(capsys):
    """
    Check if the function prints the correct output
    when there are missing values
    """
    data = pd.DataFrame({"A": [1, None, 5, 8], "B": [5, None, 5, None]})
    count_null_data(data)
    captured = capsys.readouterr()
    assert (
        captured.out.strip()
        == "Column 'A': 1 values 0\nColumn \
'B': 2 values 0"
    )


def test_no_missing_values(capsys):
    """
    Check if the function prints the correct
    output when there are no missing values
    """
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    count_null_data(data)
    captured = capsys.readouterr()
    assert captured.out.strip() == "There are no 0 value anymore!"

# Tests for delete_columns_with_zero_data function


def test_no_columns_with_zero_data():
    """
    Check if the function returns the correct output
    when there are no columns with zero data
    """
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    threshold = 0
    result = delete_columns_with_zero_data(data, threshold)
    assert result.equals(data)


def test_no_columns_dropped():
    """
    Check if the function returns the correct output
    when there is columns to be dropped
    """
    data = pd.DataFrame(
        {"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [0, 0, 0]}
    )
    threshold = 2
    expected_output = data.drop(columns=["col3"])
    assert delete_columns_with_zero_data(data, threshold).equals(
        expected_output
    )

