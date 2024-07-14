import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#matplotlib.use('Agg')  # Agg backend for non-interactive plotting

try:
    sys.path.append(os.getcwd())
    from modules.modules import *
except ModuleNotFoundError as e:
    print("You have to add the project directory to the PYTHONPATH \
           environment variable")



def main(args):
    '''this is the main function
    '''
    input_file_train = args.house_pricing_train
    data = pd.read_csv(input_file_train)
    output_dir = args.output_dir
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
    categorical_cols, numerical_cols = separate_categorical_numerical(data)
    print("Categorical columns:", categorical_cols)
    print(len(categorical_cols))
    print("Numerical columns:", numerical_cols)
    print(len(numerical_cols))

    numerical_data = data[numerical_cols].copy()

    if not os.path.exists('results'):
        os.makedirs('results')
    plt.figure(figsize=(16, 20))
    numerical_data.hist(bins=50, xlabelsize=8, ylabelsize=8)
    plt.savefig('results/plot_preprocessing/numerical_data_histogram_plot.png')
    plt.show()

    column_to_delete = ['GarageQual', 'GarageCond', 'GarageYrBlt']
    numerical_data = numerical_data.drop(column_to_delete, axis=1)

    threshold_0 = 200
    numerical_data = drop_columns_with_zero_threshold(numerical_data,
                                                      threshold_0)

    numerical_data.hist(bins=50, xlabelsize=8, ylabelsize=8)
    plt.savefig('results/plot_preprocessing/after_cleaning_numericalData_histogram_plot.png')
    plt.show()

    columns_to_transform = ['1stFlrSF', 'GrLivArea', 'LotArea', 'SalePrice']
    transformed_data = apply_1_plus_log_transformation(numerical_data,
                                                       columns_to_transform)

    transformed_data.hist(bins=50, xlabelsize=8, ylabelsize=8)
    plt.savefig('results/plot_preprocessing/transformed_data_histogram_plot.png')
    plt.show()

    plot_boxplot(numerical_data, 'OverallQual', 'SalePrice')
    plot_boxplot(numerical_data, 'YearRemodAdd', 'SalePrice')
    plot_heatmaps(transformed_data)

    X = transformed_data.iloc[:, :-1].values
    y = transformed_data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
    print("Train set shape:", X_train.shape, y_train.shape)
    print("Test set shape:", X_test.shape, y_test.shape)

    name = ['MultipleLinearRegression', 'RandomForest', 'LGBM', 'DecisionTree', 'XGB']
    model = [LinearRegression, RandomForestRegressor, LGBMRegressor, DecisionTreeRegressor, XGBRegressor]
    metrics_list = []
    models = [
        ('MultipleLinearRegression', LinearRegression()),
        ('RandomForest', RandomForestRegressor()),
        ('LGBM', LGBMRegressor()),
        ('DecisionTree', DecisionTreeRegressor()),
        ('XGB', XGBRegressor())
    ]

    param_grids = [
        {},  # No hyperparameters to tune for LinearRegression
        {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        {
            'num_leaves': [31, 50],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200]
        },
        {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
    ]
    best_models, best_params = hyperparameter_tuning(models, param_grids, X_train, y_train)

    metrics_list = []
    for name, model in best_models.items():
        output_name = f"yPred_yTrue_table_{name}.txt"
        path = f"results/evaluation_model/{output_name}"
        metrics = model_evaluation(name, model, transformed_data, path)
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    print(metrics_df)


if __name__ == '__main__':
    USAGE = 'This project is about preprocessing our dataset \
    for house pricing. therefore we need at first train.csv file \
        to train our model.'
    parser = argparse.ArgumentParser(description=USAGE)
    parser.add_argument('house_pricing_train', type=str,
                        help='Path to the house_pricing train csv file')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')
    args = parser.parse_args()
    main(args)
