import matplotlib.pyplot as plt
import seaborn as sns

def plot_categorical_columns(data):
    num_cols = len(data.columns)
    num_rows = (num_cols - 1) // 6 + 1
    fig, axes = plt.subplots(nrows=num_rows, ncols=6, figsize=(20, num_rows * 4))
    axes = axes.flatten()  # Flatten the axes array for easy indexing
    for i, column in enumerate(data.columns):
        value_counts = data[column].value_counts()
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i])
        axes[i].set_title(f'Value Counts - {column}')
        axes[i].set_xlabel('Categories')
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    # plt.show()
