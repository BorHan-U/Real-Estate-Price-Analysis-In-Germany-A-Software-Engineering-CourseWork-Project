from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError, ValidationError
import argparse
import joblib

def hyperparameter_tuning(models, param_grids, X_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV for multiple models.

    Parameters:
    models: List of tuples where each tuple contains the model name (str) and the model instance.
    param_grids: List of dictionaries with parameter names (str) as keys and lists of parameter settings to try as values.
    X_train: Training data features.
    y_train: Training data labels.

    Returns:
    best_models: Dictionary with model names as keys and the best found models as values.
    best_params: Dictionary with model names as keys and the best found parameters as values.

    Raises:
    ValueError: If models and param_grids are empty or of different lengths.
    """

    if not models or not param_grids:
        raise ValueError("The 'models' and 'param_grids' lists must not be empty.")

    if len(models) != len(param_grids):
        raise ValueError("The 'models' and 'param_grids' lists must have the same length.")

    best_models = {}
    best_params = {}

    for (name, model), param_grid in zip(models, param_grids):
        print(f"Tuning hyperparameters for {name}...")
        
        try:
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error',
                                       n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)
            
            best_models[name] = grid_search.best_estimator_
            best_params[name] = grid_search.best_params_
            
            print(f"Best parameters for {name}: {grid_search.best_params_}")
        except (ValueError, NotFittedError, ValidationError) as e:
            print(f"Error during hyperparameter tuning for {name}: {e}")
            best_models[name] = None
            best_params[name] = None

    return best_models, best_params
