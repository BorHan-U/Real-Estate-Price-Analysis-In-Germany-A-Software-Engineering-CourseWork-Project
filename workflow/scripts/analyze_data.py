"""
This module provides functions to analyze house pricing data.
It includes functionalities to generate boxplots and heatmaps based on the provided data.
"""
import argparse
import os
import sys
import pandas as pd

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.plot_boxplot import plot_boxplot
from modules.plot_heatmaps import plot_heatmaps


def analyze_data(input_file, output_dir, selected_column):
    """
    Analyze data by generating a boxplot and heatmap.
    
    Args:
        input_file (str): Path to the CSV file containing the data.
        output_dir (str): Directory where the analysis results will be saved.
        selected_column (str): Column to be used for the boxplot against SalePrice.
    """
    try:
        data = pd.read_csv(input_file)
    except pd.errors.EmptyDataError as e:
        print(f"Error reading {input_file}: {e}")
        return
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return

    # Analysis step: Generate boxplot with the specified column against 'SalePrice'
    plot_boxplot(data, selected_column, 'SalePrice', output_dir)
    
    # Heatmap by calling data from modules
    plot_heatmaps(data, output_dir)

    try:
        with open(os.path.join(output_dir, 'analysis_complete.txt'), 'w', encoding='utf-8') as file:
            file.write(f"Analysis complete. Boxplot generated for {selected_column} vs SalePrice.")
    except IOError as e:
        print(f"Error writing to file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze house pricing data.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the analysis results.")
    parser.add_argument("selected_column", type=str, help="Name of the column to plot against SalePrice.")
    args = parser.parse_args()

    analyze_data(args.input_file, args.output_dir, args.selected_column)
