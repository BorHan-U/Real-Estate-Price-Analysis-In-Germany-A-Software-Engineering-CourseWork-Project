import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplot(df, x_column, y_column, output_dir):
    data = df[[x_column, y_column]]
    
    # Print data to verify
    print(f"Plotting Boxplot for {x_column} vs {y_column}")
    print(data.describe())
    
    fig, ax = plt.subplots(figsize=(14, 9))
    sns.boxplot(x=x_column, y=y_column, data=data, ax=ax)
    plt.xticks(rotation=90)
    plt.title(f'Boxplot of {y_column} by {x_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    name = f"Boxplot_of_{y_column}_by_{x_column}.png"
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, name))
    plt.show()
