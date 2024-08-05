import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_heatmaps(df, output_dir):
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
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, "Correlation_Matrix_Heatmap.png"))
    plt.show()
