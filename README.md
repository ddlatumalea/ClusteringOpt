# ClusteringOpt

**Goal**: Using unsupervised learning techniques to cluster data.

**Datasets**: two datasets were analyzed:
- lung cancer dataset
- primates morphology dataset

## Notebooks
This project was done though the use of jupyter notebooks. The most interesting notebook is `dataset-cancer-v1.ipynb` where  Squamous Cell Carcinoma and Adenocarcinoma are analyzed. They are clustered using several methods and the performance is measured. It also contains a clustermap to show gene expression patterns. The other notebook is the morphology dataset. This dataset contains coordinates of the morphology of several species. The dataset was very hard to cluster.

## Optimizations
Throughout the notebooks I use several optimization techniques. One of those techniques is used in `functions.py`. t-SNE was also optimized through hyperparameter tuning. several models can be trained using the `optimize.sh` file. This file requires `parallel` and should be run on a linux system. By default it uses `16` cores. Be aware that t-SNE is computational heavy and will take a lot of time if the dataset is big. The `optimize.sh` file requires the path to the data and the parameter grid. These options are hardcoded. The same goes for the parameters that are given to the `optimize.py` file. The models are stored in the `output` directory and the results of the models are saved in `results.csv`. In the end, the script will only keep the best performing model (minimizing kl divergence). The `data` folder does not contain the data I used to optimize t-SNE due to the size, however they are easily generated when looking at the code.

## Plotting
For plotting the plotly.express module is used. It works great with jupyter notebooks.

--------------

**Happy investigation gene expression data and primates morphology! :)**
