"""
This module provides functionality to perform hyperparameter tuning using
GridSearchCV for multiple models.

Functions:
- hyperparameter_tuning: Perform hyperparameter tuning using GridSearchCV
  for multiple models.
- main: Parses command-line arguments and performs hyperparameter tuning
  on the specified models and parameter grids.
"""

import argparse
import joblib
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator


def hyperparameter_tuning(models, param_grids, x_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV for multiple models.

    Parameters
    ----------
    models : list of tuples
        List of tuples where each tuple contains
        the model name (str) and the model instance.
    param_grids : list of dict
        List of dictionaries with parameter names (str) as keys and
        lists of parameter settings to try as values.
    x_train : pd.DataFrame or np.ndarray
        Training data features.
    y_train : pd.Series or np.ndarray
        Training data labels.

    Returns
    -------
    best_models : dict
        Dictionary with model names as keys
        and the best found models as values.
    best_params : dict
        Dictionary with model names as keys
        and the best found parameters as values.

    Raises
    ------
    ValueError
        If models and param_grids are empty or of different lengths.
    """
    if not models or not param_grids:
        raise ValueError("The 'models' and 'param_grids'"
                         "lists must not be empty.")

    if len(models) != len(param_grids):
        raise ValueError("The 'models' and 'param_grids'"
                         "lists must have the same length.")

    best_models = {}
    best_params = {}

    for (name, model), param_grid in zip(models, param_grids):
        print(f"Tuning hyperparameters for {name}...")

        try:
            if not isinstance(model, BaseEstimator):
                raise ValueError(f"Model '{name}' is not"
                                 "a valid scikit-learn estimator.")

            grid_search = GridSearchCV(
                estimator=model, param_grid=param_grid, cv=3,
                scoring='neg_mean_squared_error', n_jobs=-1, verbose=2
            )
            grid_search.fit(x_train, y_train)

            best_models[name] = grid_search.best_estimator_
            best_params[name] = grid_search.best_params_

            print(f"Best parameters for {name}: {grid_search.best_params_}")
        except (ValueError, NotFittedError, TypeError) as exc:
            print(f"Error during hyperparameter tuning for {name}: {exc}")
            best_models[name] = None
            best_params[name] = None

    return best_models, best_params


def main():
    """
    Parses command-line arguments and performs hyperparameter tuning
    using GridSearchCV for multiple models.

    The best models and parameters are saved to joblib files.

    Raises
    ------
    SystemExit
        If the command-line arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description="Perform hyperparameter tuning using"
        "GridSearchCV for multiple models."
    )
    parser.add_argument("x_train_file", type=str,
                        help="Path to the CSV file"
                        "containing the training features.")
    parser.add_argument("y_train_file", type=str,
                        help="Path to the CSV file"
                        "containing the training labels.")
    parser.add_argument("models_file", type=str,
                        help="Path to the joblib file"
                        "containing the models to be tuned.")
    parser.add_argument("param_grids_file", type=str,
                        help="Path to the joblib file"
                        "containing the parameter grids.")
    parser.add_argument("--output_models", type=str,
                        default="best_models.joblib",
                        help="Path to save the best models.")
    parser.add_argument("--output_params", type=str,
                        default="best_params.joblib",
                        help="Path to save the best parameters.")

    args = parser.parse_args()

    try:
        # Load data
        x_train = pd.read_csv(args.x_train_file)
        y_train = pd.read_csv(args.y_train_file).squeeze()  # Convert to Series

        # Load models and parameter grids
        models = joblib.load(args.models_file)
        param_grids = joblib.load(args.param_grids_file)

        # Perform hyperparameter tuning
        best_models, best_params = hyperparameter_tuning(
            models, param_grids, x_train, y_train
            )

        # Save the best models and parameters
        joblib.dump(best_models, args.output_models)
        joblib.dump(best_params, args.output_params)

        print(f"Best models saved to {args.output_models}")
        print(f"Best parameters saved to {args.output_params}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
