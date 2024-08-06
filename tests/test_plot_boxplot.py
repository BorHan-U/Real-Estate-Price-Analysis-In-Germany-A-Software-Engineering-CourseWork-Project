"""
Unit tests for plot_boxplot module.
"""

import unittest
import pandas as pd
import sys
import os

# Add the parent directory of 'modules' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.plot_boxplot import plot_boxplot


class TestPlotBoxplot(unittest.TestCase):
    """
    Test case for the plot_boxplot function.
    """

    def test_plot(self):
        """
        Test plotting of a boxplot.
        """
        data = pd.DataFrame({
            'Category': ['A', 'B', 'A', 'B'],
            'Value': [1, 2, 3, 4]
        })
        plot_boxplot(data, 'Category', 'Value', 'output_dir')


if __name__ == '__main__':
    unittest.main()
