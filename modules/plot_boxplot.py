import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplot(df, x_column, y_column, output_dir):
    """
    Plot a boxplot of the specified columns in a DataFrame and save the plot to a file.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x_column (str): The column name to be used for the x-axis.
        y_column (str): The column name to be used for the y-axis.
        output_dir (str): The directory where the plot will be saved.

    Raises:
        ValueError: If the DataFrame is empty, the specified columns do not exist, or the y_column is not numeric.
        Exception: For any error that occurs during file writing.
    """

    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("The DataFrame is empty. Cannot plot boxplot.")
    
    # Check if specified columns exist in the DataFrame
    if x_column not in df.columns or y_column not in df.columns:
        raise ValueError(f"Columns '{x_column}' or '{y_column}' do not exist in the DataFrame.")
    
    # Check if y_column is numeric
    if not pd.api.types.is_numeric_dtype(df[y_column]):
        raise ValueError(f"The y_column '{y_column}' must be numeric to plot a boxplot.")
    
    data = df[[x_column, y_column]]
    
    # Print data summary for verification
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
    try:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, name))
        print(f"Boxplot saved as {os.path.join(output_dir, name)}")
    except Exception as e:
        raise Exception(f"Error saving the boxplot: {e}")
    
    #plt.show()  

