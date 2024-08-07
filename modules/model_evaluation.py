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
    """
    
    # Predict using the provided model
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store metrics in a dictionary
    metrics_dict = {
        'Model': name,
        'MSE': mse,
        'R2-Score': r2
    }

    # Save predictions and true values to a file
    result = np.column_stack((y_pred, y_test))  # Combine predictions and true values side by side
    with open(output_file, "w") as file:
        np.savetxt(file, result, fmt="%.2f", delimiter=",", header="Predicted,Actual", comments='')

    return metrics_dict
