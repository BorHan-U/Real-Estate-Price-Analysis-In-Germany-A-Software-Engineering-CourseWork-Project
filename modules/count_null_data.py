import pandas as pd

def count_null_data(data):
    missing_counts = (data == 0).sum()
    sorted_columns = missing_counts.sort_values(ascending=False)
    no_missing_data = True
    for column, count in sorted_columns.items():
        if pd.api.types.is_numeric_dtype(data[column]):
            nan_count = data[column].isna().sum()
            count += nan_count
        if count != 0:
            print(f"Column '{column}': {count} values 0")
            no_missing_data = False
    if no_missing_data:
        print("There are no 0 value anymore!")
