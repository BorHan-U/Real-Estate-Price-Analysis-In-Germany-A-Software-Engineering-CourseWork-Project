import os
import sys
import pandas as pd
import argparse
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

def evaluate_models(input_file, output_dir):
    data = pd.read_csv(input_file)

    # Model evaluation steps
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        ('MultipleLinearRegression', LinearRegression()),
        ('RandomForest', RandomForestRegressor()),
        ('LGBM', LGBMRegressor()),
        ('DecisionTree', DecisionTreeRegressor()),
        ('XGB', XGBRegressor())
    ]

    param_grids = [
        {},
        {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        {
            'num_leaves': [31, 50],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200]
        },
        {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
    ]

    best_models, best_params = hyperparameter_tuning(models, param_grids, X_train, y_train)

    metrics_list = []
    for name, model in best_models.items():
        output_name = f"yPred_yTrue_table_{name}.txt"
        path = f"{output_dir}/{output_name}"
        metrics = model_evaluation(name, model, data, path)
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(f"{output_dir}/metrics.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate house pricing models.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the evaluation results.")
    args = parser.parse_args()

    evaluate_models(args.input_file, args.output_dir)
