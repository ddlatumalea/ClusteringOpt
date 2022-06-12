import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import plotly.subplots as sp
import itertools

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

def express_duo_plot(figure1, figure2, score, show_legend=False):
    figure1_traces = []
    figure2_traces = []
    for trace in range(len(figure1["data"])):
        figure1_traces.append(figure1["data"][trace])
    for trace in range(len(figure2["data"])):
        figure2_traces.append(figure2["data"][trace])

    this_figure = sp.make_subplots(rows=1, cols=2, subplot_titles=('Prediction', 'Truth')) 

    for traces in figure1_traces:
        this_figure.append_trace(traces, row=1, col=1)
    for traces in figure2_traces:
        this_figure.append_trace(traces, row=1, col=2)

    this_figure.update_layout(title_text=f'Adjusted rand score of {score:.2f}', showlegend=False)
    return this_figure

def rand_score_and_plot(figure1, figure2, y_true, y_pred, show_legend=False):    
    score = adjusted_rand_score(y_true, y_pred)
    fig = express_duo_plot(figure1, figure2, score, show_legend=show_legend)

    return fig

def create_param_options(param_dict):
    grid = []
    keys = list(param_dict.keys())
    values = list(param_dict.values())
    value_combs = list(itertools.product(*values))

    for val in value_combs:
        temp = dict(zip(keys, val))
        grid.append(temp)

    return grid

def optimize(estimator, X, y, param_options):
    best_score = 0
    best_model = None
    best_pred = None

    combinations = create_param_options(param_options)
    param_grid = pd.DataFrame(combinations)
    header = param_grid.columns.values
    
    n = param_grid.shape[0]
    
    for i in range(param_grid.shape[0]):
        print(f'Evaluation combination {i+1}/{n}.')
        param_vals = param_grid.iloc[i].values.tolist()
        params = dict(zip(header, param_vals))
        try:
            model = estimator(**params)
            y_pred = model.fit_predict(X)
            score = adjusted_rand_score(y, y_pred)
        
            if score > best_score:
                best_score = score
                best_model = model
                best_pred = y_pred
        except Exception as e:
            print(e)
            
    return best_score, best_model, best_pred