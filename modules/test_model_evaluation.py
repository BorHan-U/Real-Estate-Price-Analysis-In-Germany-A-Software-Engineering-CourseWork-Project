import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from modules.modules import model_evaluation
from sklearn.metrics import mean_squared_error, r2_score
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="[LightGBM] [Warning] No further splits with positive gain, best gain: -inf")


class TestModelEvaluation(unittest.TestCase):

    def generate_test_data(self, num_samples=100):
        # Generate synthetic data with a linear relationship and some noise
        np.random.seed(42)
        data = np.random.rand(num_samples, 5)
        target = 2 * data[:, 0] + 3 * data[:, 1] + np.random.normal(loc=0, scale=0.1, size=num_samples)

        # Convert the data to DataFrame format
        data = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
        data['target'] = target
        return data
