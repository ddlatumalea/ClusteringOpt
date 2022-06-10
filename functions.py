import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_distribution(data, rows, cols):
    """Plots the distribution of the given data with the mean and median.
    
    Keyword arguments:
    data -- a pandas DataFrame containing the data.
    """
    fig, axs = plt.subplots(rows, cols, figsize=(15, 15), constrained_layout=True)

    for ax, col in zip(axs.flatten(), data.columns):
        mean = data[col].mean()
        median = data[col].median()

        sns.histplot(data=data, x=col, ax=ax)
        ax.axvline(mean, ls='--', color='red', alpha=0.7)
        ax.axvline(median, ls='--', color='purple', alpha=0.7)

        min_ylim, max_ylim = ax.get_ylim()
        ax.text(mean*1.1, max_ylim*0.9, 'Mean: {:.4f}'.format(mean))
        ax.text(median*1.1, max_ylim*0.7, 'Median: {:.4f}'.format(median)) 
        
    plt.show()
    
def scree_plot(pca: PCA):
    x = np.arange(pca.n_components_) + 1
    
    plt.plot(x, pca.explained_variance_ratio_, 'o-', color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal component')
    plt.ylabel('Explained variance')
    
    plt.show()
    
def component_plot(pca_comp1, pca_comp2, y, title):
    pc1, xlabel = pca_comp1
    pc2, ylabel = pca_comp2
    
    plot = sns.scatterplot(x=pc1, y=pc2, hue=y, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.show()
   
    return plot