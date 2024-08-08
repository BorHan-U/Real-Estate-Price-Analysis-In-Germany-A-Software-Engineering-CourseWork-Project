import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def model_evaluation(name, model, X_test, y_test, output_file):
    """
    Evaluate a model and save the predicted and actual values to a file.

    Args:
        name (str): Name of the model.
        model: Trained model to be evaluated.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): True values for the test set.
        output_file (str): Path to the file where predictions and true values will be saved.

    Returns:
        dict: Dictionary containing evaluation metrics.
    
    Raises:
        ValueError: If X_test or y_test is empty, or if their shapes are incompatible.
        Exception: For any error that occurs during model prediction or file writing.
    """
    
    # Check if X_test and y_test are empty
    if X_test.size == 0 or y_test.size == 0:
        raise ValueError("X_test and y_test must not be empty.")
    
    # Check if the shapes of X_test and y_test are compatible
    if len(X_test) != len(y_test):
        raise ValueError("The number of samples in X_test and y_test must be the same.")
    
    try:
        # Predict using the provided model
        y_pred = model.predict(X_test)
    except Exception as e:
        raise Exception(f"Error during model prediction: {e}")

    try:
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    except ValueError as e:
        raise ValueError(f"Error calculating evaluation metrics: {e}")
    
    # Store metrics in a dictionary
    metrics_dict = {
        'Model': name,
        'MSE': mse,
        'R2-Score': r2
    }

    try:
        # Save predictions and true values to a file
        result = np.column_stack((y_pred, y_test))  # Combine predictions and true values side by side
        with open(output_file, "w") as file:
            np.savetxt(file, result, fmt="%.2f", delimiter=",", header="Predicted,Actual", comments='')
    except Exception as e:
        raise Exception(f"Error writing results to file: {e}")

    return metrics_dict
