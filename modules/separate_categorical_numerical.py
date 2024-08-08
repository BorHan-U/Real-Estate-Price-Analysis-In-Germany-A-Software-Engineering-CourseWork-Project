import pandas as pd

def separate_categorical_numerical(data):
    """
    Separates the columns of a DataFrame into categorical and numerical columns.

    Args:
        data (pd.DataFrame): The DataFrame to be processed.

    Returns:
        tuple: A tuple containing two lists - categorical columns and numerical columns.

    Raises:
        ValueError: If the DataFrame is empty or contains no columns.
    """
    
    # Check if DataFrame is empty
    if data.empty:
        raise ValueError("The DataFrame is empty. Cannot separate columns.")
    
    # Check if DataFrame has no columns
    if len(data.columns) == 0:
        raise ValueError("The DataFrame contains no columns.")
    
    categorical_cols = []
    numerical_cols = []
    
    for column in data.columns:
        dtype = data[column].dtype
        
        if pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            categorical_cols.append(column)
        elif pd.api.types.is_numeric_dtype(dtype):
            numerical_cols.append(column)
        else:
            print(f"Column '{column}' has an unrecognized data type: {dtype}. It will be ignored.")

    return categorical_cols, numerical_cols
