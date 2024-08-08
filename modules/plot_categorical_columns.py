import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_categorical_columns(data):
    """
    Plots bar charts for the value counts of each categorical column in the DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.

    Raises:
        ValueError: If the DataFrame is empty or if all columns are non-categorical.
    """
    
    if data.empty:
        raise ValueError("The DataFrame is empty. Cannot plot categorical columns.")
    
    # Filter out non-categorical columns and columns with too many unique values
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    categorical_columns = [col for col in categorical_columns if data[col].nunique() <= 20]

    if len(categorical_columns) == 0:
        raise ValueError("No categorical columns with a reasonable number of unique values to plot.")
    
    num_cols = len(categorical_columns)
    num_rows = (num_cols - 1) // 6 + 1
    fig, axes = plt.subplots(nrows=num_rows, ncols=6, figsize=(20, num_rows * 4))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i, column in enumerate(categorical_columns):
        value_counts = data[column].value_counts()
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i])
        axes[i].set_title(f'Value Counts - {column}')
        axes[i].set_xlabel('Categories')
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=45)
    
    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    # plt.show()
