"""
This module contains unit tests for the delete_columns_with_zero_data function.

The goal of these tests is to ensure robust
Quality Assurance by testing the function's
behavior with various inputs, including edge cases
and unexpected data types. This
comprehensive testing approach helps verify that
the function handles different
scenarios correctly and maintains data integrity.

Classes:
    TestDeleteColumnsWithZeroData: A test case class
    for the delete_columns_with_zero_data function.
"""

import unittest
from typing import Any
import pandas as pd
import numpy as np

from modules.delete_columns_with_zero_data import delete_columns_with_zero_data


class TestDeleteColumnsWithZeroData(unittest.TestCase):
    """
    A test case class for the delete_columns_with_zero_data function.

    This class contains various test methods
    to ensure the correct functionality
    of the delete_columns_with_zero_data
    function under different scenarios,
    including normal operations, edge cases, and error handling.
    """

    def test_delete_columns(self) -> None:
        """
        Test the basic functionality of deleting
        columns with zero data above a threshold.

        This test verifies that the function
        correctly removes columns where the number
        of zero values exceeds the specified threshold.
        """
        data = pd.DataFrame({
            'A': [0, 0, 0],
            'B': [1, 2, 3],
            'C': [0, 1, 0]
        })
        threshold = 2
        result = delete_columns_with_zero_data(data, threshold)
        expected_data = pd.DataFrame({
            'B': [1, 2, 3],
            'C': [0, 1, 0]
        })
        pd.testing.assert_frame_equal(result, expected_data)

    def test_empty_dataframe(self) -> None:
        """
        Test the handling of an empty DataFrame.

        This test ensures that the function raises
        a ValueError when given an empty DataFrame.
        """
        data = pd.DataFrame()
        threshold = 2
        with self.assertRaises(ValueError):
            delete_columns_with_zero_data(data, threshold)

    def test_negative_threshold(self) -> None:
        """
        Test the handling of a negative threshold.

        This test verifies that the function raises
        a ValueError when given a negative threshold.
        """
        data = pd.DataFrame({'A': [1, 2, 3]})
        threshold = -1
        with self.assertRaises(ValueError):
            delete_columns_with_zero_data(data, threshold)

    def test_non_dataframe_input(self) -> None:
        """
        Test the handling of non-DataFrame input.

        This test ensures that the function raises
        a TypeError when given input that is not a DataFrame.
        """
        data: Any = [1, 2, 3]
        threshold = 2
        with self.assertRaises(TypeError):
            delete_columns_with_zero_data(data, threshold)

    def test_non_integer_threshold(self) -> None:
        """
        Test the handling of a non-integer threshold.

        This test verifies that the function raises
        a TypeError when given a non-integer threshold.
        """
        data = pd.DataFrame({'A': [1, 2, 3]})
        threshold = 2.5
        with self.assertRaises(TypeError):
            delete_columns_with_zero_data(data, threshold)

    def test_mixed_data_types(self) -> None:
        """
        Test the handling of mixed data types.

        This test ensures that the function
        correctly processes DataFrames with mixed data types,
        including numeric, string, and boolean columns.
        """
        data = pd.DataFrame({
            'A': [0, 0, 0],
            'B': ['a', 'b', 'c'],
            'C': [1, 2, 3],
            'D': [True, False, True]
        })
        threshold = 2
        result = delete_columns_with_zero_data(data, threshold)
        expected_data = pd.DataFrame({
            'B': ['a', 'b', 'c'],
            'C': [1, 2, 3],
            'D': [True, False, True]
        })
        pd.testing.assert_frame_equal(result, expected_data)

    def test_nan_values(self) -> None:
        """
        Test the handling of NaN values.

        This test verifies that the function
        treats NaN values similarly to zero values
        when determining which columns to drop.
        """
        data = pd.DataFrame({
            'A': [0, np.nan, 0],
            'B': [1, 2, 3],
            'C': [np.nan, np.nan, np.nan]
        })
        threshold = 2
        result = delete_columns_with_zero_data(data, threshold)
        expected_data = pd.DataFrame({
            'B': [1, 2, 3]
        })
        pd.testing.assert_frame_equal(result, expected_data)

    def test_all_columns_dropped(self) -> None:
        """
        Test the case where all columns are dropped.

        This test ensures that the function
        correctly handles the scenario where
        all columns in the DataFrame exceed
        the threshold and should be dropped.
        """
        data = pd.DataFrame({
            'A': [0, 0, 0],
            'B': [0, np.nan, 0],
            'C': [np.nan, np.nan, np.nan]
        })
        threshold = 1
        result = delete_columns_with_zero_data(data, threshold)
        # Check that the result has no columns
        self.assertEqual(len(result.columns), 0)
        # Check that the result has the same number of rows as the input
        self.assertEqual(len(result), len(data))
        # Check that the result has the same index as the input
        pd.testing.assert_index_equal(result.index, data.index)

    def test_no_columns_dropped(self) -> None:
        """
        Test the case where no columns are dropped.

        This test verifies that the function
        correctly handles the scenario where
        no columns in the DataFrame exceed
        the threshold and thus none are dropped.
        """
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        threshold = 0
        result = delete_columns_with_zero_data(data, threshold)
        pd.testing.assert_frame_equal(result, data)


if __name__ == '__main__':
    unittest.main()
