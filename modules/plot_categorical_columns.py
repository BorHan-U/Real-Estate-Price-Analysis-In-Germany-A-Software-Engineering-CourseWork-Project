import matplotlib.pyplot as plt
import seaborn as sns

def plot_categorical_columns(data):
    num_cols = len(data.columns)
    num_rows = (num_cols - 1) // 6 + 1
    fig, axes = plt.subplots(nrows=num_rows, ncols=6, figsize=(20, num_rows * 4))
    for i, column in enumerate(data.columns):
        row = i // 6
        col = i % 6
        value_counts = data[column].value_counts()
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[row, col])
        axes[row, col].set_title(f'Value Counts - {column}')
        axes[row, col].set_xlabel('Categories')
        axes[row, col].set_ylabel('Count')
        axes[row, col].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
