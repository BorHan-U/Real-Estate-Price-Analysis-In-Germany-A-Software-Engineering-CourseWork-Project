import numpy as np
import pandas as pd

def apply_1_plus_log_transformation(data, columns_to_transform):
    """
    Applies the log(1 + x) transformation to specified columns in a DataFrame.
    
    Parameters:
    data (pd.DataFrame): The input data as a pandas DataFrame.
    columns_to_transform (list): A list of column names to apply the transformation to.
    
    Returns:
    pd.DataFrame: The transformed data with log(1 + x) applied to specified columns.
    
    Raises:
    ValueError: If any column in columns_to_transform is not in the DataFrame.
    """

    transformed_data = data.copy()

    for column in columns_to_transform:
        if column not in transformed_data.columns:
            raise ValueError(f"Column '{column}' is not in the DataFrame.")
        try:
            # Handle non-numeric data types gracefully
            transformed_data[column] = pd.to_numeric(transformed_data[column], errors='coerce')
            # Apply the log(1 + x) transformation
            transformed_data[column] = np.log1p(transformed_data[column])
        except Exception as e:
            print(f"Error applying log(1 + x) transformation to column '{column}': {e}")
            raise

    return transformed_data
