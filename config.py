PATHS = {
    'dataset_dir': 'data',
    'labels': 'data/dataset/labels_strict_cobra.csv',
    'roi_series': 'data/dataset/COBRA_ALL_only_strict.npz'
}

DATASET_PARAMS = {
    'graph_dataset': {
        'neighbors': 32
    },
}

MODEL_PARAMS = {
    'hidden_dim': 128
}