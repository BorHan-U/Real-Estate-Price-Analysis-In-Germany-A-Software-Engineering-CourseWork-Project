import pandas as pd

def separate_categorical_numerical(data):
    categorical_cols = []
    numerical_cols = []
    for column in data.columns:
        if data[column].dtype == 'object' or isinstance(data[column].dtype, pd.CategoricalDtype):
            categorical_cols.append(column)
        else:
            numerical_cols.append(column)
    return categorical_cols, numerical_cols
