"""
Unit tests for separate_categorical_numerical module.
"""

import unittest
import pandas as pd
import sys
import os

# Add the parent directory of 'modules' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.separate_categorical_numerical import separate_categorical_numerical


class TestSeparateCategoricalNumerical(unittest.TestCase):
    """
    Test case for the separate_categorical_numerical function.
    """

    def test_separate(self):
        """
        Test separation of categorical and numerical columns.
        """
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['X', 'Y', 'Z'],
            'C': [4.5, 5.5, 6.5]
        })
        categorical_cols, numerical_cols = separate_categorical_numerical(data)
        self.assertEqual(categorical_cols, ['B'])
        self.assertEqual(numerical_cols, ['A', 'C'])


if __name__ == '__main__':
    unittest.main()
