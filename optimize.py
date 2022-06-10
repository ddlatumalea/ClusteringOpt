import argparse
import pickle
from typing import Any
import pandas as pd
import sys
import random

from sklearn.manifold import TSNE

OUTPUT = 'output/'
MODELS = ['TSNE']

class TSNEOptimizer:

    def __init__(self, param_dict) -> None:
        self.param_grid = param_dict
        self.id = random.randint(0, 1000000)

    def fit(self, X) -> None:
        model = TSNE(**self.param_grid)
        _ = model.fit_transform(X)

        self.kl_div = model.kl_divergence_
        self.model = model

    def write(self):
        # save model
        pickle.dump(self.model, open(f'{OUTPUT}{self.id}.pkl', 'wb'))
        # write to stdout
        sys.stdout.write(f'{self.id},{self.kl_div}\n')

def is_float(element: Any) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False



if __name__ == '__main__':
    """
    
    Example:
    python optimize.py --model TSNE --params p1 p2 p3 --header h1 h2 h3
    """
    parser = argparse.ArgumentParser(description='Requires the input parameters for grid search.')
    parser.add_argument('--model', help='Model to optimize', required=True, type=str)
    parser.add_argument('--params', nargs='+', help='Parameters', required=True)
    parser.add_argument('--header', nargs='+', help='Parameter heads', required=True)
    parser.add_argument('--data', help='Path to data to fit to', required=True)

    args = parser.parse_args()

    m = args.model
    params = args.params
    header = args.header
    data_path = args.data

    if len(params) != len(header):
        raise ValueError("Expects header to be of same legnth as parameters.")

    X = pd.read_csv(data_path)
    
    # create an option for standardizing etc.

    arguments = dict(zip(header, params))

    for k, v in arguments.items():
        try:
            if v.isnumeric():
                arguments[k] = int(v)
            elif is_float(v):
                arguments[k] = float(v)
        except ValueError:
            arguments[k] = str(v)

    if m == 'TSNE':
        model = TSNEOptimizer(arguments)
        model.fit(X)
        model.write()

    else:
        raise ValueError('Model not known. Choose one of the following models:', MODELS)

