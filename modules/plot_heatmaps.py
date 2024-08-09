import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import argparse

def plot_heatmaps(df, output_dir):
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    corrmat = numeric_df.corr()

    f, ax = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot the full correlation matrix heatmap
    sns.heatmap(corrmat, vmax=0.8, square=True, cmap="RdBu", ax=ax[0])
    ax[0].set_title('Correlation Matrix Heatmap')

    k = 10
    if 'SalePrice' in corrmat.columns:
        cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
        cm = np.corrcoef(df[cols].values.T)
        sns.set(font_scale=1.25)
        sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                    annot_kws={'size': 10}, yticklabels=cols.values,
                    xticklabels=cols.values, cmap="RdBu", ax=ax[1])
        ax[1].set_title('Top 10 most correlated variables with sale price')
    else:
        ax[1].set_visible(False)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the heatmap as a PNG file
    plt.savefig(os.path.join(output_dir, "Correlation_Matrix_Heatmap.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot correlation matrix heatmaps from a DataFrame and save them to a file.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file containing the data.")
    parser.add_argument("output_dir", type=str, help="Directory where the heatmaps will be saved.")
    
    args = parser.parse_args()

    # Load the data from the CSV file
    df = pd.read_csv(args.input_file)

    # Plot the heatmaps
    plot_heatmaps(df, args.output_dir)

if __name__ == "__main__":
    main()
