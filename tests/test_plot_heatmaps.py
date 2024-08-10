"""
Unit tests for plot_heatmaps module.

This module contains comprehensive tests to ensure the correct functionality
of the plot_heatmaps function under various scenarios, including
edge cases and error handling.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
import tempfile
import shutil

# Set the matplotlib backend to 'Agg' for non-interactive plotting
import matplotlib
matplotlib.use('Agg')

# Add the parent directory of 'modules' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.plot_heatmaps import plot_heatmaps, PlotSaveError


class TestPlotHeatmaps(unittest.TestCase):
    """
    Test case for the plot_heatmaps function.

    This class contains various test methods to ensure the correct functionality
    of the plot_heatmaps function under different scenarios, including
    normal operations, edge cases, and error handling.
    """

    def setUp(self):
        """Set up test data and temporary directory."""
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': [1, 3, 5, 2, 4],
            'SalePrice': [100, 200, 150, 300, 250]
        })
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_plot_normal(self):
        """Test plotting of heatmaps with normal input."""
        plot_heatmaps(self.data, self.temp_dir)
        expected_file = os.path.join(self.temp_dir, "Correlation_Matrix_Heatmap.png")
        self.assertTrue(os.path.exists(expected_file))

    def test_empty_dataframe(self):
        """Test handling of an empty DataFrame."""
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            plot_heatmaps(empty_df, self.temp_dir)

    def test_non_numeric_dataframe(self):
        """Test handling of a DataFrame with no numeric columns."""
        non_numeric_df = pd.DataFrame({
            'A': ['a', 'b', 'c'],
            'B': ['d', 'e', 'f']
        })
        with self.assertRaises(ValueError):
            plot_heatmaps(non_numeric_df, self.temp_dir)

    def test_without_saleprice(self):
        """Test plotting without 'SalePrice' column."""
        data_without_saleprice = self.data.drop('SalePrice', axis=1)
        plot_heatmaps(data_without_saleprice, self.temp_dir)
        expected_file = os.path.join(self.temp_dir, "Correlation_Matrix_Heatmap.png")
        self.assertTrue(os.path.exists(expected_file))

    def test_single_column(self):
        """Test plotting with a single numeric column."""
        single_column_df = pd.DataFrame({'A': [1, 2, 3]})
        plot_heatmaps(single_column_df, self.temp_dir)
        expected_file = os.path.join(self.temp_dir, "Correlation_Matrix_Heatmap.png")
        self.assertTrue(os.path.exists(expected_file))

    def test_large_dataframe(self):
        """Test plotting with a large DataFrame."""
        large_df = pd.DataFrame(np.random.rand(1000, 50))
        plot_heatmaps(large_df, self.temp_dir)
        expected_file = os.path.join(self.temp_dir, "Correlation_Matrix_Heatmap.png")
        self.assertTrue(os.path.exists(expected_file))

    def test_output_dir_creation(self):
        """Test creation of output directory if it doesn't exist."""
        new_dir = os.path.join(self.temp_dir, 'new_subdir')
        plot_heatmaps(self.data, new_dir)
        expected_file = os.path.join(new_dir, "Correlation_Matrix_Heatmap.png")
        self.assertTrue(os.path.exists(expected_file))


if __name__ == '__main__':
    unittest.main()
