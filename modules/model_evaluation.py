"""
This module provides functionality to evaluate a trained model and save
the predicted and actual values to a file.

Functions:
- model_evaluation: Evaluate a model and save the predicted and actual values to a file.
- main: Parses command-line arguments and evaluates the specified model.
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


class ModelEvaluationError(Exception):
    """Custom exception for errors during model evaluation."""

def model_evaluation(name, model, x_test, y_test, output_file):
    """
    Evaluate a model and save the predicted and actual values to a file.

    Parameters
    ----------
    name : str
        Name of the model.
    model : object
        Trained model to be evaluated.
    x_test : np.ndarray
        Test features.
    y_test : np.ndarray
        True values for the test set.
    output_file : str
        Path to the file where predictions and true values will be saved.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.

    Raises
    ------
    ValueError
        If x_test or y_test is empty, or if their shapes are incompatible.
    ModelEvaluationError
        For any error that occurs during model prediction or file writing.
    """
    if x_test.size == 0 or y_test.size == 0:
        raise ValueError("x_test and y_test must not be empty.")

    if len(x_test) != len(y_test):
        raise ValueError("The number of samples in x_test and y_test must be the same.")

    try:
        # Predict using the provided model
        y_pred = model.predict(x_test)
    except Exception as exc:
        raise ModelEvaluationError(f"Error during model prediction: {exc}") from exc

    try:
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    except ValueError as exc:
        raise ValueError(f"Error calculating evaluation metrics: {exc}") from exc

    # Store metrics in a dictionary
    metrics_dict = {
        'Model': name,
        'MSE': mse,
        'R2-Score': r2
    }

    try:
        # Save predictions and true values to a file
        result = np.column_stack((y_pred, y_test))
        with open(output_file, "w", encoding="utf-8") as file:
            np.savetxt(file, result, fmt="%.2f", delimiter=",",
                       header="Predicted,Actual", comments='')
    except Exception as exc:
        raise ModelEvaluationError(f"Error writing results to file: {exc}") from exc

    return metrics_dict


def main():
    """
    Parses command-line arguments and evaluates a trained model.

    The evaluation metrics and predictions are saved to specified files.

    Raises
    ------
    SystemExit
        If the command-line arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model and save predictions with true values."
    )
    parser.add_argument(
        "model_file", type=str,
        help="Path to the trained model file (joblib format)."
    )
    parser.add_argument(
        "x_test_file", type=str,
        help="Path to the CSV file containing the test features."
    )
    parser.add_argument(
        "y_test_file", type=str,
        help="Path to the CSV file containing the true test values."
    )
    parser.add_argument(
        "output_file", type=str,
        help="Path to the output file where predictions and true values will be saved."
    )
    parser.add_argument(
        "--model_name", type=str, default="Model",
        help="Name of the model being evaluated."
    )

    args = parser.parse_args()

    # Load the model
    model = joblib.load(args.model_file)

    # Load the test data
    x_test = pd.read_csv(args.x_test_file).values
    y_test = pd.read_csv(args.y_test_file).values.flatten()

    # Evaluate the model
    metrics = model_evaluation(args.model_name, model, x_test, y_test, args.output_file)

    # Print evaluation metrics
    print(f"Evaluation metrics for {args.model_name}:")
    print(f"MSE: {metrics['MSE']}")
    print(f"R2-Score: {metrics['R2-Score']}")


if __name__ == "__main__":
    main()
