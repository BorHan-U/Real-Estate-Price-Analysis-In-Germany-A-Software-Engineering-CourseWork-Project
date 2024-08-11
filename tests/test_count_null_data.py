"""
This module contains unit tests for the count_null_data function.

It tests various scenarios including normal DataFrames, empty DataFrames,
DataFrames with all NaN values, non-DataFrame inputs, and mixed data types.
"""

import unittest
import io
from contextlib import redirect_stdout
import pandas as pd
import numpy as np
from modules.count_null_data import count_null_data


class TestCountNullData(unittest.TestCase):
    """
    Test case for the count_null_data function.

    This class contains various test methods to ensure the correct
    functionality of the count_null_data function under different scenarios.
    """

    def setUp(self):
        """
        Set up test cases.

        This method is called before each test. It creates a standard
        DataFrame for use in multiple tests.
        """
        self.data_normal = pd.DataFrame({
            'A': [0, 1, 2, np.nan],
            'B': [0, 0, 0, 1],
            'C': [1, 2, 3, 4],
            'D': ['a', 'b', 'c', 'd']
        })

    def capture_output(self, func, *args, **kwargs):
        """
        Capture the stdout output of a function.

        Args:
            func (callable): The function to capture output from.
            *args: Variable length argument list for func.
            **kwargs: Arbitrary keyword arguments for func.

        Returns:
            str: The captured stdout output.
        """
        f = io.StringIO()
        with redirect_stdout(f):
            func(*args, **kwargs)
        return f.getvalue()

    def test_count_null_data_normal(self):
        """
        Test counting of null data in a normal DataFrame.

        This test checks if the function correctly identifies and counts
        zero and NaN values in a DataFrame with mixed data.
        """
        result = count_null_data(self.data_normal)
        self.assertEqual(result['A']['total'], 2)
        self.assertEqual(result['B']['total'], 3)
        self.assertNotIn('C', result)
        self.assertEqual(result['D'], 'skipped')

    def test_count_null_data_empty(self):
        """
        Test counting of null data in an empty DataFrame.

        This test verifies that the function
        handles empty DataFrames correctly.
        """
        data_empty = pd.DataFrame()
        result = count_null_data(data_empty)
        self.assertEqual(result, {})

    def test_count_null_data_all_nan(self):
        """
        Test counting of null data in a DataFrame with all NaN values.

        This test checks if the function correctly counts NaN values when
        all values in the DataFrame are NaN.
        """
        data_all_nan = pd.DataFrame({'A': [np.nan, np.nan],
                                     'B': [np.nan, np.nan]})
        result = count_null_data(data_all_nan)
        self.assertEqual(result['A']['total'], 2)
        self.assertEqual(result['B']['total'], 2)

    def test_count_null_data_non_dataframe(self):
        """
        Test counting of null data with non-DataFrame input.

        This test verifies that the function raises a TypeError when
        the input is not a pandas DataFrame.
        """
        with self.assertRaises(TypeError):
            count_null_data([1, 2, 3])

    def test_count_null_data_mixed_types(self):
        """
        Test counting of null data in a DataFrame with mixed types.

        This test checks if the function correctly handles a DataFrame
        with various data types including numeric, string, and boolean.
        """
        data_mixed = pd.DataFrame({
            'A': [0, 1, 2, np.nan],
            'B': ['a', 'b', 'c', 'd'],
            'C': [True, False, True, False],
            'D': [1.0, 2.0, np.nan, 4.0]
        })
        result = count_null_data(data_mixed)
        self.assertEqual(result['A']['total'], 2)
        self.assertEqual(result['B'], 'skipped')
        self.assertEqual(result['C'], 'skipped')
        self.assertEqual(result['D']['total'], 1)


if __name__ == '__main__':
    unittest.main()
