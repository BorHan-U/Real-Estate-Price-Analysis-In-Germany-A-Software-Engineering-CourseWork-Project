import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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

def delete_columns_with_zero_data(data, threshold):
    for column in data.columns:
        zero_count = (data[column] == 0).sum()
        if zero_count > threshold:
            data = data.drop(column, axis=1)
    return data


def separate_categorical_numerical(data):
    categorical_cols = []
    numerical_cols = []
    for column in data.columns:
        if data[column].dtype == 'object' or pd.api.types.\
                            is_categorical_dtype(data[column].dtype):
            categorical_cols.append(column)
        else:
            numerical_cols.append(column)
    return categorical_cols, numerical_cols
