import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import os
import sys
import argparse


'''adding the project directory to the
PYTHONPATH environment variables and
importing the modules script
'''

try:
    sys.path.append(os.getcwd())
    from modules.modules import *
except ModuleNotFoundError as e:
    print("You have to add the project directory to the PYTHONPATH \
           environment variable")
    


def main(args):

    input_file_train = args.house_pricing_train
    data = pd.read_csv(input_file_train)

    data = data.drop('Id', axis=1)

    print("The shape of our dataset is: ", data.shape)
    print()
    data.info()

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


if __name__ == '__main__':
    USAGE = 'This project is about preprocessing our dataset \
    for house pricing. therefore we need at first train.csv file \
        to train our model, then test it using the test.csv file.'
    parser = argparse.ArgumentParser(description=USAGE)
    parser.add_argument('house_pricing_train', type=str,
                        help='Path to the house_pricing train csv file')
    args = parser.parse_args()
    main(args)
