PATHS = {
    'dataset_dir': 'data',
    'labels': 'data/dataset/labels_strict_cobra.csv',
    'roi_series': 'data/dataset/COBRA_ALL_only_strict.npz'
}

DATASET_PARAMS = {
    'graph_dataset': {
        'neighbors': 32
    },
    'BrainNetCNN': {
        'load_dir': 'data/dataset/schiza-norm.mat'
    }
}

MODEL_PARAMS = {
    'base': {
        'hidden_dim': 128
    },
    'BrainNetCNN': {
    }
}

SEED = 322