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