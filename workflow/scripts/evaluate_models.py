"""
The script evaluate different machine learning models
such Linear Regression and Emsemble learning with
hyperameter tuning to find the best params
and best model to fit the data.
"""
import argparse
import logging
import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Add the root directory to the Python path
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '../..')))

from modules.hyperparameter_tuning import hyperparameter_tuning
from modules.model_evaluation import model_evaluation


# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_param_grids():
    """
    Return a dictionary of models and their
    corresponding hyperparameter grids.

    Returns:
        dict: A dictionary where keys are model
        names and values are hyperparameter grids.
    """
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
    """
    Evaluate models using the provided dataset and save the results.

    Args:
        input_file (str): Path to the input CSV file.
        output_dir (str): Directory to save the evaluation results.
    """
    if not os.path.isfile(input_file):
        logging.error("Input file '%s' does not exist.", input_file)
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info("Created output directory '%s'.", output_dir)

    data = pd.read_csv(input_file)
    logging.info("Loaded data from '%s' with shape '%s'.",
                 input_file, data.shape)

    x_train, x_test, y_train, y_test = split_data(data)
    models = get_models()
    param_grids = get_param_grids()
    best_models, best_params = hyperparameter_tuning(
        models, [param_grids[name] for name, _ in models], x_train, y_train)

    log_best_params(best_params)
    metrics_list = evaluate_and_save_models(
        best_models, x_test, y_test, output_dir)
    save_metrics(metrics_list, output_dir)
    save_best_params(best_params, output_dir)


def split_data(data):
    """
    Split the data into training and testing sets.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        tuple: x_train, x_test, y_train, y_test
    """
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)
    logging.info("Split data into train and test sets with "
                 "shapes '%s' and '%s'.", x_train.shape, x_test.shape)
    return x_train, x_test, y_train, y_test


def get_models():
    """
    Return a list of models to be evaluated.

    Returns:
        list: A list of tuples where each tuple contains
        a model name and an instance of the model.
    """
    return [
        ('MultipleLinearRegression', LinearRegression()),
        ('RandomForest', RandomForestRegressor()),
        ('LGBM', LGBMRegressor()),
        ('DecisionTree', DecisionTreeRegressor()),
        ('XGB', XGBRegressor())
    ]


def log_best_params(best_params):
    """
    Log the best hyperparameters for each model.

    Args:
        best_params (dict): A dictionary where keys are
        model names and values are the best hyperparameters.
    """
    for name, params in best_params.items():
        logging.info("Best parameters for '%s': '%s'", name, params)


def evaluate_and_save_models(best_models, x_test, y_test, output_dir):
    """
    Evaluate the best models and save the results.

    Args:
        best_models (dict): A dictionary where keys are
        model names and values are the best model instances.
        x_test (np.ndarray): The test features.
        y_test (np.ndarray): The test labels.
        output_dir (str): Directory to save the evaluation results.

    Returns:
        list: A list of evaluation metrics for each model.
    """
    metrics_list = []
    for name, model in best_models.items():
        output_name = f"yPred_yTrue_table_{name}.txt"
        path = os.path.join(output_dir, output_name)
        metrics = model_evaluation(name, model, x_test, y_test, path)
        metrics_list.append(metrics)
        logging.info("Evaluated model '%s' and saved results to '%s'.",
                     name, path)
    return metrics_list


def save_metrics(metrics_list, output_dir):
    """
    Save the evaluation metrics to a CSV file.

    Args:
        metrics_list (list): A list of evaluation metrics for each model.
        output_dir (str): Directory to save the evaluation results.
    """
    metrics_df = pd.DataFrame(metrics_list)
    metrics_csv_path = os.path.join(output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    logging.info("Saved evaluation metrics to '%s'.", metrics_csv_path)


def save_best_params(best_params, output_dir):
    """
    Save the best hyperparameters to a CSV file.

    Args:
        best_params (dict): A dictionary where keys are
        model names and values are the best hyperparameters.
        output_dir (str): Directory to save the evaluation results.
    """
    best_params_df = pd.DataFrame.from_dict(best_params, orient='index')
    best_params_csv_path = os.path.join(output_dir, "best_params.csv")
    best_params_df.to_csv(best_params_csv_path)
    logging.info("Saved best hyperparameters to '%s'.", best_params_csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate house pricing models.")
    parser.add_argument(
        "input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument("output_dir", type=str,
                        help="Directory to save the evaluation results.")
    args = parser.parse_args()

    evaluate_models(args.input_file, args.output_dir)
