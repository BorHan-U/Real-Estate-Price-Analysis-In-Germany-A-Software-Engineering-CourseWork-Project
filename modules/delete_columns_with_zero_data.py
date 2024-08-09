import pandas as pd
import argparse

def delete_columns_with_zero_data(data, threshold):
    """
    Deletes columns from a DataFrame where the number of zero values exceeds a given threshold.
    
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
    
    columns_to_drop = []
    
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            zero_count = (data[column] == 0).sum()
            if zero_count > threshold:
                columns_to_drop.append(column)
        else:
            print(f"Column '{column}' is not numeric and was skipped.")
    
    if columns_to_drop:
        data = data.drop(columns=columns_to_drop)
        print(f"Dropped columns: {', '.join(columns_to_drop)}")
    else:
        print("No columns were dropped.")
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Delete columns from a DataFrame where zero values exceed a given threshold.")
    parser.add_argument("file", type=str, help="Path to the input CSV file.")
    parser.add_argument("threshold", type=int, help="Threshold for the maximum allowed number of zero values in a column.")
    parser.add_argument("--output", type=str, default="filtered_data.csv", help="Path to save the filtered CSV file.")

    args = parser.parse_args()

    # Read the data from the CSV file
    data = pd.read_csv(args.file)

    # Apply the column deletion based on zero values
    filtered_data = delete_columns_with_zero_data(data, args.threshold)

    # Save the filtered data to a CSV file
    filtered_data.to_csv(args.output, index=False)
    print(f"Filtered data saved to {args.output}")

if __name__ == "__main__":
    main()
