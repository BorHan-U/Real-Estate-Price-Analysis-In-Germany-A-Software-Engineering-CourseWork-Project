"""
Unit tests for plot_categorical_columns module.

This module contains comprehensive tests to ensure the correct functionality
of the plot_categorical_columns function under various scenarios, including
edge cases and error handling.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
import tempfile
import shutil
from unittest.mock import patch

# Set the matplotlib backend to 'Agg' for non-interactive plotting
import matplotlib
matplotlib.use('Agg')

# Add the parent directory of 'modules' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.plot_categorical_columns import plot_categorical_columns, PlotSaveError


class TestPlotCategoricalColumns(unittest.TestCase):
    """
    Test case for the plot_categorical_columns function.

    This class contains various test methods to ensure the correct functionality
    of the plot_categorical_columns function under different scenarios, including
    normal operations, edge cases, and error handling.
    """

    def setUp(self):
        """Set up test data and temporary directory."""
        self.data = pd.DataFrame({
            'Category1': ['A', 'B', 'A', 'B', 'C', 'C'],
            'Category2': ['X', 'Y', 'X', 'Y', 'Z', 'Z'],
            'Numeric': [1, 2, 3, 4, 5, 6],
            'ManyCategories': [f'Cat{i}' for i in range(6)]
        })
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_plot_normal(self):
        """Test plotting of categorical columns with normal input."""
        plot_categorical_columns(self.data, self.temp_dir)
        expected_file = os.path.join(self.temp_dir, "categorical_columns_plots.png")
        self.assertTrue(os.path.exists(expected_file))

    def test_empty_dataframe(self):
        """Test handling of an empty DataFrame."""
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            plot_categorical_columns(empty_df, self.temp_dir)

    def test_no_categorical_columns(self):
        """Test handling of a DataFrame with no categorical columns."""
        numeric_df = pd.DataFrame({
            'Num1': [1, 2, 3],
            'Num2': [4, 5, 6]
        })
        with self.assertRaises(ValueError):
            plot_categorical_columns(numeric_df, self.temp_dir)

    def test_too_many_categories(self):
        """Test handling of columns with too many unique values."""
        many_categories_df = pd.DataFrame({
            'ManyCategories': [f'Cat{i}' for i in range(30)]
        })
        with self.assertRaises(ValueError):
            plot_categorical_columns(many_categories_df, self.temp_dir)

    @patch('matplotlib.pyplot.savefig')
    def test_plot_save_error(self, mock_savefig):
        """Test handling of plot save error."""
        mock_savefig.side_effect = OSError("Mocked OSError")
        with self.assertRaises(PlotSaveError):
            plot_categorical_columns(self.data, self.temp_dir)


    def test_no_output_dir(self):
        """Test plotting without specifying an output directory."""
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_categorical_columns(self.data)
            mock_show.assert_called_once()

    def test_mixed_data_types(self):
        """Test handling of mixed data types."""
        mixed_df = pd.DataFrame({
            'Category': ['A', 'B', 'C'],
            'Numeric': [1, 2, 3],
            'Boolean': [True, False, True]
        })
        plot_categorical_columns(mixed_df, self.temp_dir)
        expected_file = os.path.join(self.temp_dir, "categorical_columns_plots.png")
        self.assertTrue(os.path.exists(expected_file))

    def test_output_dir_creation(self):
        """Test creation of output directory if it doesn't exist."""
        new_dir = os.path.join(self.temp_dir, 'new_subdir')
        plot_categorical_columns(self.data, new_dir)
        expected_file = os.path.join(new_dir, "categorical_columns_plots.png")
        self.assertTrue(os.path.exists(expected_file))


if __name__ == '__main__':
    unittest.main()
