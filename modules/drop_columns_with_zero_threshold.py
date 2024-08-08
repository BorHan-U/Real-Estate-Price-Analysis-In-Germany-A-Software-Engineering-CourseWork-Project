import pandas as pd

def drop_columns_with_zero_threshold(data, threshold):
    """
    Drops columns from the DataFrame where the number of zero values exceeds the given threshold.
    
    Parameters:
    data (pd.DataFrame): The input data as a pandas DataFrame.
    threshold (int): The maximum allowed number of zero values in a column before it is dropped.
    
    Returns:
    pd.DataFrame: The DataFrame with columns removed where zero values exceed the threshold.
    
    Raises:
    ValueError: If the DataFrame is empty or the threshold is negative.
    """

    if data.empty:
        raise ValueError("The DataFrame is empty.")
    
    if threshold < 0:
        raise ValueError("Threshold must be a non-negative integer.")
    
    # Ensure we only consider numeric columns
    numeric_data = data.select_dtypes(include=[int, float, bool])
    
    if numeric_data.empty:
        print("No numeric columns found. No columns were dropped.")
        return data
    
    zero_counts = (numeric_data == 0).sum()
    columns_to_drop = zero_counts[zero_counts > threshold].index
    
    if columns_to_drop.empty:
        print("No columns have zero values exceeding the threshold. No columns were dropped.")
    else:
        data = data.drop(columns=columns_to_drop)
        print(f"Dropped columns: {', '.join(columns_to_drop)}")
    
    return data
