from model.simple_model import GraphNetwork
from model.brainnetcnn import BrainNetCNN

MODEL_NAME = 'BrainNetCNN'
DATASET_NAME = 'abide'


PATHS = {
    'dataset_dir': 'data',
    'labels': 'data/dataset/labels_strict_cobra.csv',
    'roi_series': 'data/dataset/COBRA_ALL_only_strict.npz'
}


SPLITS = {
        'cobre': 'data/data_base/cobre/cobre_splits.json',
        'abide': 'data/data_base/abide/abide_splits.json',
    }


DATASET_PARAMS = {
    'graph_dataset': {
        'neighbors': 32
    },

    'BrainNetCNN': {
        'cobre': {
            'load_dir': 'data/data_base/cobre/',
            'conn_matr_name': 'schiza-norm.mat'
        },
        'abide': {
            'load_dir': 'data/data_base/abide/',
            'conn_matr_name': 'schiza-norm.mat'
        },
    }
}


MODEL_CLASSES = {
    'base' : GraphNetwork,
    'BrainNetCNN': BrainNetCNN,
}


MODEL_PARAMS = {
    'base': {
        'hidden_dim': 128
    },
    'BrainNetCNN': {
    }
}

SEED = 322