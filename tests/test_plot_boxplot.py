"""
Unit tests for plot_boxplot module.

This module contains comprehensive tests to ensure the correct functionality
of the plot_boxplot function under various scenarios, including
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

from modules.plot_boxplot import plot_boxplot, PlotSaveError


class TestPlotBoxplot(unittest.TestCase):
    """
    Test case for the plot_boxplot function.

    This class contains various test methods to ensure the correct functionality
    of the plot_boxplot function under different scenarios, including
    normal operations, edge cases, and error handling.
    """

    def setUp(self):
        """Set up test data and temporary directory."""
        self.data = pd.DataFrame({
            'Category': ['A', 'B', 'A', 'B', 'C', 'C'],
            'Value': [1, 2, 3, 4, 5, 6],
            'NonNumeric': ['X', 'Y', 'Z', 'X', 'Y', 'Z']
        })
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_plot_normal(self):
        """Test plotting of a boxplot with normal input."""
        plot_boxplot(self.data, 'Category', 'Value', self.temp_dir)
        expected_file = os.path.join(self.temp_dir, 'Boxplot_of_Value_by_Category.png')
        self.assertTrue(os.path.exists(expected_file))

    def test_empty_dataframe(self):
        """Test handling of an empty DataFrame."""
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            plot_boxplot(empty_df, 'Category', 'Value', self.temp_dir)

    def test_nonexistent_column(self):
        """Test handling of non-existent columns."""
        with self.assertRaises(ValueError):
            plot_boxplot(self.data, 'NonExistent', 'Value', self.temp_dir)

    def test_non_numeric_y_column(self):
        """Test handling of non-numeric y_column."""
        with self.assertRaises(ValueError):
            plot_boxplot(self.data, 'Category', 'NonNumeric', self.temp_dir)

    @patch('matplotlib.pyplot.savefig')
    def test_plot_save_error(self, mock_savefig):
        """Test handling of plot save error."""
        mock_savefig.side_effect = OSError("Mocked OSError")
        with self.assertRaises(PlotSaveError):
            plot_boxplot(self.data, 'Category', 'Value', self.temp_dir)

    def test_single_category(self):
        """Test plotting with a single category."""
        single_category_data = pd.DataFrame({
            'Category': ['A'] * 5,
            'Value': range(1, 6)
        })
        plot_boxplot(single_category_data, 'Category', 'Value', self.temp_dir)
        expected_file = os.path.join(self.temp_dir, 'Boxplot_of_Value_by_Category.png')
        self.assertTrue(os.path.exists(expected_file))

    def test_many_categories(self):
        """Test plotting with many categories."""
        many_categories_data = pd.DataFrame({
            'Category': [f'Cat{i}' for i in range(50)],
            'Value': np.random.rand(50)
        })
        plot_boxplot(many_categories_data, 'Category', 'Value', self.temp_dir)
        expected_file = os.path.join(self.temp_dir, 'Boxplot_of_Value_by_Category.png')
        self.assertTrue(os.path.exists(expected_file))

    def test_non_string_category(self):
        """Test plotting with non-string category values."""
        non_string_category_data = pd.DataFrame({
            'Category': [1, 2, 3, 1, 2, 3],
            'Value': range(1, 7)
        })
        plot_boxplot(non_string_category_data, 'Category', 'Value', self.temp_dir)
        expected_file = os.path.join(self.temp_dir, 'Boxplot_of_Value_by_Category.png')
        self.assertTrue(os.path.exists(expected_file))

    def test_output_dir_creation(self):
        """Test creation of output directory if it doesn't exist."""
        new_dir = os.path.join(self.temp_dir, 'new_subdir')
        plot_boxplot(self.data, 'Category', 'Value', new_dir)
        expected_file = os.path.join(new_dir, 'Boxplot_of_Value_by_Category.png')
        self.assertTrue(os.path.exists(expected_file))


if __name__ == '__main__':
    unittest.main()
