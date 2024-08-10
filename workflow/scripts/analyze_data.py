import os
import random
import sys
import pandas as pd
import argparse

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.plot_boxplot import plot_boxplot
from modules.plot_heatmaps import plot_heatmaps

def analyze_data(input_file, output_dir):
    data = pd.read_csv(input_file)

    # Randomly select a column for the boxplot
    columns = data.columns.tolist()
    columns.remove('SalePrice')  # Exclude 'SalePrice' itself
    selected_column = random.choice(columns)
    
    # Analysis step: Generate dynamic boxplots with the random columns against 'SalePrice'
    plot_boxplot(data, selected_column, 'SalePrice', output_dir)
    
    with open(f'{output_dir}/analysis_complete.txt', 'w') as f:
        f.write(f"Analysis complete. Boxplot generated for {selected_column} vs SalePrice.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze house pricing data.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the analysis results.")
    args = parser.parse_args()

    analyze_data(args.input_file, args.output_dir)
