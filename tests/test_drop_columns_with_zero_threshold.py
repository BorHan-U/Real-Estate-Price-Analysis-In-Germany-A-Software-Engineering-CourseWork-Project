"""
Unit tests for drop_columns_with_zero_threshold module.
"""

import unittest
import pandas as pd
import sys
import os

# Add the parent directory of 'modules' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.drop_columns_with_zero_threshold import drop_columns_with_zero_threshold


class TestDropColumnsWithZeroThreshold(unittest.TestCase):
    """
    Test case for the drop_columns_with_zero_threshold function.
    """

    def test_drop_columns(self):
        """
        Test dropping of columns with zero data above a threshold.
        """
        data = pd.DataFrame({
            'A': [0, 0, 0],
            'B': [1, 2, 3],
            'C': [0, 1, 0]
        })
        threshold = 2
        result = drop_columns_with_zero_threshold(data, threshold)
        expected_data = pd.DataFrame({
            'B': [1, 2, 3],
            'C': [0, 1, 0]
        })
        pd.testing.assert_frame_equal(result, expected_data)


if __name__ == '__main__':
    unittest.main()
