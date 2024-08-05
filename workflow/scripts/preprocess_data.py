import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.count_null_data import count_null_data
from modules.delete_columns_with_zero_data import delete_columns_with_zero_data
from modules.separate_categorical_numerical import separate_categorical_numerical
from modules.drop_columns_with_zero_threshold import drop_columns_with_zero_threshold
from modules.apply_1_plus_log_transformation import apply_1_plus_log_transformation

def preprocess_data(input_file, output_file, output_dir):
    data = pd.read_csv(input_file)

    # Preprocessing steps
    data = data.drop('Id', axis=1)

    mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    columns_to_map = ['GarageQual', 'GarageCond', 'PoolQC', 'FireplaceQu',
                      'KitchenQual', 'HeatingQC', 'BsmtCond', 'BsmtQual',
                      'ExterCond', 'ExterQual']

    for column in columns_to_map:
        data[column] = data[column].map(mapping)

    count_null_data(data)

    columns = data.columns.tolist()
    columns.insert(-1, 'Age')
    data['Age'] = data['YrSold'] - data['YearBuilt']
    columns.remove('YearBuilt')
    columns.remove('YrSold')
    data = data[columns]

    data = data.fillna(0)

    count_null_data(data)

    threshold = 900
    data = delete_columns_with_zero_data(data, threshold)

    count_null_data(data)
    categorical_cols, numerical_cols = separate_categorical_numerical(data)

    numerical_data = data[numerical_cols].copy()

    plt.figure(figsize=(16, 20))
    numerical_data.hist(bins=50, xlabelsize=8, ylabelsize=8)
    plt.savefig(f'{output_dir}/numerical_data_histogram_plot.png')
    plt.close()

    column_to_delete = ['GarageQual', 'GarageCond', 'GarageYrBlt']
    numerical_data = numerical_data.drop(column_to_delete, axis=1)

    threshold_0 = 200
    numerical_data = drop_columns_with_zero_threshold(numerical_data, threshold_0)

    numerical_data.hist(bins=50, xlabelsize=8, ylabelsize=8)
    plt.savefig(f'{output_dir}/after_cleaning_numericalData_histogram_plot.png')
    plt.close()

    columns_to_transform = ['1stFlrSF', 'GrLivArea', 'LotArea', 'SalePrice']
    transformed_data = apply_1_plus_log_transformation(numerical_data, columns_to_transform)

    transformed_data.hist(bins=50, xlabelsize=8, ylabelsize=8)
    plt.savefig(f'{output_dir}/transformed_data_histogram_plot.png')
    plt.close()

    # Save the preprocessed data
    transformed_data.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess house pricing data.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument("output_file", type=str, help="Path to save the preprocessed CSV file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the plots.")
    args = parser.parse_args()

    preprocess_data(args.input_file, args.output_file, args.output_dir)
