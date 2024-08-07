"""
Unit tests for apply_1_plus_log_transformation module.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory of 'modules' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.apply_1_plus_log_transformation import apply_1_plus_log_transformation


class TestApply1PlusLogTransformation(unittest.TestCase):
    """
    Test case for the apply_1_plus_log_transformation function.
    """

    def test_transformation(self):
        """
        Test the log transformation on specified columns.
        """
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        columns_to_transform = ['A', 'B']
        transformed_data = apply_1_plus_log_transformation(data, columns_to_transform)
        expected_data = pd.DataFrame({
            'A': np.log1p([1, 2, 3]),
            'B': np.log1p([4, 5, 6])
        })
        pd.testing.assert_frame_equal(transformed_data, expected_data)


if __name__ == '__main__':
    unittest.main()
