import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

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

def drop_columns_with_zero_threshold(data, threshold):
    zero_counts = (data == 0).sum()
    columns_to_drop = zero_counts[zero_counts > threshold].index
    data = data.drop(columns=columns_to_drop)
    print(zero_counts)
    return data


def plot_categorical_columns(data):
    num_cols = len(data.columns)
    num_rows = (num_cols - 1) // 6 + 1
    fig, axes = plt.subplots(nrows=num_rows, ncols=6,
                             figsize=(20, num_rows * 4))
    for i, column in enumerate(data.columns):
        row = i // 6
        col = i % 6
        value_counts = data[column].value_counts()
        sns.barplot(x=value_counts.index, y=value_counts.values,
                    ax=axes[row, col])
        axes[row, col].set_title(f'Value Counts - {column}')
        axes[row, col].set_xlabel('Categories')
        axes[row, col].set_ylabel('Count')
        axes[row, col].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()


def apply_1_plus_log_transformation(data, columns_to_transform):
    transformed_data = data.copy()
    for column in columns_to_transform:
        transformed_data[column] = np.log1p(transformed_data[column])
    return transformed_data


def model_evaluation(name, model, data, output_file):

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics_dict = {
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'R2-Score': r2
    }

    result = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
    with open(output_file, "w") as file:
        np.savetxt(file, result, fmt="%.2f", delimiter=",")

    return metrics_dict


def plot_boxplot(df, x_column, y_column):

    data = df[[x_column, y_column]]
    fig, ax = plt.subplots(figsize=(14, 9))
    sns.boxplot(x=x_column, y=y_column, data=data, ax=ax)
    ax.set_ylim(0, 800000)
    plt.xticks(rotation=90)
    plt.title(f'Boxplot of {y_column} by {x_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    name = f"Boxplot of {y_column} by {x_column}.png"
    plt.savefig(f"results/plot_preprocessing/{name}.png")
    plt.show()


def plot_heatmaps(df):
    corrmat = df.corr()
    f, ax = plt.subplots(1, 2, figsize=(20, 10))
    sns.heatmap(corrmat, vmax=0.8, square=True, cmap="RdBu", ax=ax[0])
    ax[0].set_title('Correlation Matrix Heatmap')
    k = 10
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.25)
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                annot_kws={'size': 10}, yticklabels=cols.values,
                xticklabels=cols.values, cmap="RdBu", ax=ax[1])
    ax[1].set_title('Top 10 most correlated variables with sale price')
    plt.savefig("results/plot_preprocessing/Correlation Matrix Heatmap.png")
    plt.show()
    

def hyperparameter_tuning(models, param_grids, X_train, y_train):
    """
        Perform hyperparameter tuning using GridSearchCV for multiple models.
        Parameters:
        models: List of tuples where each tuple contains the model name (str) and the model instance.
        param_grids: List of dictionaries with parameter names (str) as keys and lists of parameter settings to try as values.
        X_train: Training data features.
        y_train: Training data labels.
        Returns:
        best_models: Dictionary with model names as keys and the best found models as values.
        best_params: Dictionary with model names as keys and the best found parameters as values.
    """
    best_models = {}
    best_params = {}

    for (name, model), param_grid in zip(models, param_grids):
        print(f"Tuning hyperparameters for {name}...")
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error',
                                   n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        best_models[name] = grid_search.best_estimator_
        best_params[name] = grid_search.best_params_

        print(f"Best parameters for {name}: {grid_search.best_params_}")

    return best_models, best_params

