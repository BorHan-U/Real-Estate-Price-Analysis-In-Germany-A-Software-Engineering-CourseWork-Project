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


# Tests for separate_categorical_numerical function


def test_numerical_columns():
    """
    Check if the function returns the correct output
    when there are only numerical columns
    """
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    categorical_cols, numerical_cols = separate_categorical_numerical(data)
    assert categorical_cols == []
    assert numerical_cols == ["col1", "col2"]


def test_only_categorical_columns():
    """
    Check if the function returns the correct output
    when there are only categorical columns
    """
    data = pd.DataFrame({"col1": ["a", "b", "c"], "col2": ["d", "e", "f"]})
    categorical_cols, numerical_cols = separate_categorical_numerical(data)
    assert categorical_cols == ["col1", "col2"]
    assert numerical_cols == []


def test_separate_categorical_numerical():
    """
    Check if the function returns the correct output
    when there are both categorical and numerical columns
    """
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    categorical_cols, numerical_cols = separate_categorical_numerical(data)
    assert categorical_cols == ["col2"]
    assert numerical_cols == ["col1"]


def test_empty_dataframe():
    """
    Check if the function returns the correct output
    when the dataframe is empty
    """
    data = pd.DataFrame()
    categorical_cols, numerical_cols = separate_categorical_numerical(data)
    assert categorical_cols == []
    assert numerical_cols == []


# Tests for drop_columns_with_zero_threshold function


def test_no_zero_values():
    """
    Check if the function returns the correct output
    when there are no zero values
    """
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    threshold = 1
    result = drop_columns_with_zero_threshold(data, threshold)
    assert result.equals(data)


def test_drop_columns_with_zero_threshold_with_threshold_0():
    """
    Check if the function drops the columns when
    there are columns with zero values and threshold is 0
    """
    data = pd.DataFrame({"A": [1, 2, 3], "B": [0, 0, 0], "C": [0, 0, 0]})
    threshold = 0
    expected_output = pd.DataFrame({"A": [1, 2, 3]})
    assert drop_columns_with_zero_threshold(data, threshold).equals(
        expected_output
    )


def test_threshold_greater_than_zero_counts():
    """
    Check if the function returns the correct output when
    there are columns with zero values and threshold is greater
    than amount of zero values
    """
    data = pd.DataFrame({"A": [1, 2, 3], "B": [0, 0, 0], "C": [0, 0, 0]})
    threshold = 3
    expected_output = data
    output = drop_columns_with_zero_threshold(data, threshold)
    pd.testing.assert_frame_equal(output, expected_output)


def test_threshold_equal_to_one_column_zero_count():
    """
    Check if the function returns the correct output when
    there are columns with zero values and threshold
    is equal to amount of zero values
    """
    data = pd.DataFrame(
        {"col1": [0, 1, 2], "col2": [0, 0, 0], "col3": [1, 2, 3]}
    )
    threshold = 2
    expected_output = pd.DataFrame({"col1": [0, 1, 2], "col3": [1, 2, 3]})
    assert drop_columns_with_zero_threshold(data, threshold).equals(
        expected_output
    )


# Tests for apply_1_plus_log_transformation function


def test_single_column_transformation():
    """
    Check if the function returns the correct output
    when there is only one column to be transformed
    """
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    columns_to_transform = ["col1"]
    transformed_data = apply_1_plus_log_transformation(
        data, columns_to_transform
    )
    assert transformed_data["col1"].tolist() == [
        np.log1p(1),
        np.log1p(2),
        np.log1p(3),
    ]
    assert transformed_data["col2"].tolist() == [4, 5, 6]


def test_multiple_columns_transformation():
    """
    Check if the function returns the correct output
    when there are multiple columns to be transformed
    """
    data = pd.DataFrame(
        {"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]}
    )
    columns_to_transform = ["col1", "col3"]
    transformed_data = apply_1_plus_log_transformation(
        data.copy(), columns_to_transform
    )
    assert transformed_data["col1"].equals(np.log1p(data["col1"]))
    assert transformed_data["col2"].equals(data["col2"])
    assert transformed_data["col3"].equals(np.log1p(data["col3"]))
