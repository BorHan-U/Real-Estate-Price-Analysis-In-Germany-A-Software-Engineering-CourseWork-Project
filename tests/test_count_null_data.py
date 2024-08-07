"""
Unit tests for count_null_data module.
"""

import unittest
import pandas as pd
import sys
import os

# Add the parent directory of 'modules' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.count_null_data import count_null_data


class TestCountNullData(unittest.TestCase):
    """
    Test case for the count_null_data function.
    """

    def test_count_null_data(self):
        """
        Test counting of null data in the DataFrame.
        """
        data = pd.DataFrame({
            'A': [0, 1, 2],
            'B': [0, 0, 0],
            'C': [1, 2, 3]
        })
        count_null_data(data)


if __name__ == '__main__':
    unittest.main()
