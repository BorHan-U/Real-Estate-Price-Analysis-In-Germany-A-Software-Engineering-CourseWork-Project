import os
import sys
import pandas as pd
import argparse
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.hyperparameter_tuning import hyperparameter_tuning
from modules.model_evaluation import model_evaluation

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_param_grids():
    """Return a dictionary of models and their corresponding hyperparameter grids."""
    return {
        'MultipleLinearRegression': {},
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'LGBM': {
            'num_leaves': [31, 50],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200]
        },
        'DecisionTree': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'XGB': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
    }

def evaluate_models(input_file, output_dir):
    """Evaluate models using the provided dataset and save the results."""
    # Validate input file and output directory
    if not os.path.isfile(input_file):
        logging.error(f"Input file '{input_file}' does not exist.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory '{output_dir}'.")

    # Load data
    data = pd.read_csv(input_file)
    logging.info(f"Loaded data from '{input_file}' with shape {data.shape}.")

    # Split data
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Split data into train and test sets with shapes {X_train.shape} and {X_test.shape}.")

    # Model definitions
    models = [
        ('MultipleLinearRegression', LinearRegression()),
        ('RandomForest', RandomForestRegressor()),
        ('LGBM', LGBMRegressor()),
        ('DecisionTree', DecisionTreeRegressor()),
        ('XGB', XGBRegressor())
    ]

    # Get parameter grids
    param_grids = get_param_grids()

    # Hyperparameter tuning
    best_models, best_params = hyperparameter_tuning(models, [param_grids[name] for name, _ in models], X_train, y_train)

    # Log best parameters
    for name, params in best_params.items():
        logging.info(f"Best parameters for {name}: {params}")

    # Model evaluation
    metrics_list = []
    for name, model in best_models.items():
        output_name = f"yPred_yTrue_table_{name}.txt"
        path = os.path.join(output_dir, output_name)
        metrics = model_evaluation(name, model, X_test,y_test, path)
        metrics_list.append(metrics)
        logging.info(f"Evaluated model '{name}' and saved results to '{path}'.")

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_list)
    metrics_csv_path = os.path.join(output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    logging.info(f"Saved evaluation metrics to '{metrics_csv_path}'.")

    # Save best parameters to CSV
    best_params_df = pd.DataFrame.from_dict(best_params, orient='index')
    best_params_csv_path = os.path.join(output_dir, "best_params.csv")
    best_params_df.to_csv(best_params_csv_path)
    logging.info(f"Saved best hyperparameters to '{best_params_csv_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate house pricing models.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the evaluation results.")
    args = parser.parse_args()

    evaluate_models(args.input_file, args.output_dir)
