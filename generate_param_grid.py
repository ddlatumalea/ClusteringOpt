import argparse
import json
import pandas as pd
import numpy as np
import itertools as it

def create_param_options(param_dict):
    grid = []
    keys = list(param_dict.keys())
    values = list(param_dict.values())
    value_combs = list(it.product(*values))

    for val in value_combs:
        temp = dict(zip(keys, val))
        grid.append(temp)

    return grid

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', help='Path to input json to read parameters', required=True)
	parser.add_argument('-o', help='Path to save search grid to.', required=True)

	args = parser.parse_args()

	json_path = args.p
	output_path = args.o

	with open(json_path, 'r') as f:
		params = json.load(f)

	combinations = create_param_options(params)
	param_grid = pd.DataFrame(combinations)
	
	param_grid.to_csv(output_path, index=False)