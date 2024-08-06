"""
Unit tests for plot_categorical_columns module.
"""

import unittest
import pandas as pd
import sys
import os

# Add the parent directory of 'modules' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.plot_categorical_columns import plot_categorical_columns


class TestPlotCategoricalColumns(unittest.TestCase):
    """
    Test case for the plot_categorical_columns function.
    """

    def test_plot(self):
        """
        Test plotting of categorical columns.
        """
        data = pd.DataFrame({
            'Category1': ['A', 'B', 'A', 'B'],
            'Category2': ['X', 'Y', 'X', 'Y']
        })
        plot_categorical_columns(data)


if __name__ == '__main__':
    unittest.main()
