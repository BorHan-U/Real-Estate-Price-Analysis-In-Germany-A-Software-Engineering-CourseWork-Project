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

    def test_model_evaluation(self):
        # Generate test data
        test_data = self.generate_test_data()

        # Define models to test
        models = [
            ('Multiple Linear Regression', LinearRegression),
            ('Random Forest', RandomForestRegressor),
            ('LGBM', LGBMRegressor)
        ]

        for name, model_class in models:
            # Evaluate the model
            output_file = f"results/evaluation_model/yPred_yTrue_table_{name}_testCase.txt"
            metrics_dict = model_evaluation(name, model_class, test_data, output_file)

            # Print the results
            print(f"Model: {metrics_dict['Model']}")
            print(f"MSE: {metrics_dict['MSE']}")
            print(f"R2-Score: {metrics_dict['R2-Score']}")
            print()

            # Check if MSE and R-squared are within the desired range
            self.assertLess(metrics_dict['MSE'], 0.1)
            self.assertGreater(metrics_dict['R2-Score'], 0.8)


if __name__ == '__main__':
    unittest.main()
