"""
Unit tests for separate_categorical_numerical module.

This module contains comprehensive tests to ensure the correct functionality
of the separate_categorical_numerical function under various scenarios,
including edge cases and error handling.
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import json
from io import StringIO
from unittest.mock import patch
from modules.separate_categorical_numerical import separate_categorical_numerical, main


class TestSeparateCategoricalNumerical(unittest.TestCase):
    """
    Test case for the separate_categorical_numerical function and main function.

    This class contains various test methods to ensure the correct functionality
    of the separate_categorical_numerical function and main function under different scenarios,
    including normal operations, edge cases, and error handling.
    """

    def test_separate_mixed_types(self):
        """Test separation of mixed data types."""
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['X', 'Y', 'Z'],
            'C': [4.5, 5.5, 6.5],
            'D': [True, False, True]
        })
        numerical_cols = separate_categorical_numerical(data)
        self.assertEqual(set(numerical_cols), {'A', 'C'})

    def test_all_numerical(self):
        """Test with all numerical columns."""
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4.5, 5.5, 6.5],
            'C': [7, 8, 9]
        })
        numerical_cols = separate_categorical_numerical(data)
        self.assertEqual(set(numerical_cols), {'A', 'B', 'C'})


    def test_empty_dataframe(self):
        """Test with an empty DataFrame."""
        data = pd.DataFrame()
        with self.assertRaises(ValueError):
            separate_categorical_numerical(data)

    def test_no_columns(self):
        """Test with a DataFrame that has no columns."""
        data = pd.DataFrame(index=[0, 1, 2])
        with self.assertRaises(ValueError):
            separate_categorical_numerical(data)

    def test_single_column(self):
        """Test with a single column DataFrame."""
        data = pd.DataFrame({'A': [1, 2, 3]})
        numerical_cols = separate_categorical_numerical(data)
        self.assertEqual(numerical_cols, ['A'])

    def test_mixed_numeric_types(self):
        """Test with mixed numeric types (int and float)."""
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4.5, 5.5, 6.5],
            'C': np.array([7, 8, 9], dtype=np.int64)
        })
        numerical_cols = separate_categorical_numerical(data)
        self.assertEqual(set(numerical_cols), {'A', 'B', 'C'})

    def test_categorical_as_numeric(self):
        """Test with categorical data represented as numeric."""
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': pd.Categorical([1, 2, 3]),
            'C': [4.5, 5.5, 6.5]
        })
        numerical_cols = separate_categorical_numerical(data)
        self.assertEqual(set(numerical_cols), {'A', 'C'})

    def test_datetime_column(self):
        """Test with a datetime column."""
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': pd.date_range('2023-01-01', periods=3),
            'C': [4.5, 5.5, 6.5]
        })
        numerical_cols = separate_categorical_numerical(data)
        self.assertEqual(set(numerical_cols), {'A', 'C'})

    @patch('sys.stdout', new_callable=StringIO)
    def test_unrecognized_dtype(self, mock_stdout):
        """Test with an unrecognized data type."""
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [complex(1, 2), complex(3, 4), complex(5, 6)],
            'C': [4.5, 5.5, 6.5]
        })
        numerical_cols = separate_categorical_numerical(data)
        self.assertEqual(set(numerical_cols), {'A', 'C'})
        self.assertIn("Column 'B' has an unrecognized data type", mock_stdout.getvalue())

    def test_main_function(self):
        """Test the main function."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_csv:
            temp_csv.write("A,B,C\n1,X,4.5\n2,Y,5.5\n3,Z,6.5\n")
            temp_csv_name = temp_csv.name

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_num:
            temp_num_name = temp_num.name

        test_args = [
            'separate_categorical_numerical.py',
            temp_csv_name,
            '--output_numerical', temp_num_name
        ]
        with patch('sys.argv', test_args):
            main()

        with open(temp_num_name, 'r') as f:
            num_cols = json.load(f)
        self.assertEqual(set(num_cols), {'A', 'C'})
        os.unlink(temp_csv_name)
        os.unlink(temp_num_name)


if __name__ == '__main__':
    unittest.main()
