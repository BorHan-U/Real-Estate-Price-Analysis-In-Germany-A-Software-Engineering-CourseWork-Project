"""
Unit tests for delete_columns_with_zero_data module.
"""

import unittest
import pandas as pd
import sys
import os

# Add the parent directory of 'modules' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.delete_columns_with_zero_data import delete_columns_with_zero_data


class TestDeleteColumnsWithZeroData(unittest.TestCase):
    """
    Test case for the delete_columns_with_zero_data function.
    """

    def test_delete_columns(self):
        """
        Test deletion of columns with zero data above a threshold.
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


if __name__ == '__main__':
    unittest.main()
