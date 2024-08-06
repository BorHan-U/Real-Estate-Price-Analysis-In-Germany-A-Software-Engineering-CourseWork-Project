"""
Unit tests for plot_heatmaps module.
"""

import unittest
import pandas as pd
import sys
import os

# Add the parent directory of 'modules' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.plot_heatmaps import plot_heatmaps


class TestPlotHeatmaps(unittest.TestCase):
    """
    Test case for the plot_heatmaps function.
    """

    def test_plot(self):
        """
        Test plotting of heatmaps.
        """
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        plot_heatmaps(data, 'output_dir')


if __name__ == '__main__':
    unittest.main()
