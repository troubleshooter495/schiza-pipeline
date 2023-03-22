from torch_geometric.data import InMemoryDataset
import torch
import numpy as np
import pandas as pd
from typing import Dict, List

from torch_geometric.utils import from_networkx
from networkx.convert_matrix import from_numpy_matrix
from nilearn.connectome import ConnectivityMeasure

import torch.utils.data.dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from utils.io import read_file_locally, read_data
from config import PATHS, DATASET_PARAMS, SEED

import os


class GraphDataset(InMemoryDataset):
    def __init__(self, root: str, transform=None, pre_transform=None, neighbors: int = 10):
        self.neighbors = neighbors
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        """ Converts raw data into GNN-readable format by constructing
        graphs out of connectivity matrices.
        """
        graphs = []
        labels = read_file_locally(PATHS['labels'])
        roi_series = read_file_locally(PATHS['roi_series'])
        labels = torch.from_numpy(labels.Dx.to_numpy())
        corr_matrices, pcorr_matrices = self._get_corr_mats(roi_series['sub'])
        assert len(labels) == len(corr_matrices) == len(pcorr_matrices)

        for i in range(0, len(corr_matrices)):
            pcorr_matrix_np = pcorr_matrices[i]

            index = np.abs(pcorr_matrix_np).argsort(axis=1)
            n_rois = pcorr_matrix_np.shape[0]

            for j in range(n_rois):
                for k in range(n_rois - self.neighbors):
                    pcorr_matrix_np[j, index[j, k]] = 0
                for k in range(n_rois - self.neighbors, n_rois):
                    pcorr_matrix_np[j, index[j, k]] = 1

            pcorr_matrix_nx = from_numpy_matrix(pcorr_matrix_np)
            pcorr_matrix_data = from_networkx(pcorr_matrix_nx)

            corr_matrix_np = corr_matrices[i]

            pcorr_matrix_data.x = torch.tensor(corr_matrix_np).float()
            pcorr_matrix_data.y = labels[i].type(torch.LongTensor)
            graphs.append(pcorr_matrix_data)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])

    def _get_corr_mats(self, roi_series: np.array):
        corr_measure = ConnectivityMeasure(kind='correlation')
        pcorr_measure = ConnectivityMeasure(kind='partial correlation')
        corr_matrices = corr_measure.fit_transform(roi_series)
        pcorr_matrices = pcorr_measure.fit_transform(roi_series)
        return corr_matrices, pcorr_matrices


class BrainNetDataset(torch.utils.data.Dataset):
    def __init__(self, save_dir: str, load_dir: str, conn_matr_name: np.array = None,
                 mode: str = 'train', kfold_labels: List[str] = None, transform=False):
        """
        Args:
            load_dir (string): Path to the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.directory = load_dir
        self.transform = transform
        self.mode = mode

        if len(conn_matr_name):
            conn_matr = read_file_locally(load_dir + conn_matr_name)
            x = conn_matr['dti']
            y_all = conn_matr['label'][0]
            ids = conn_matr['id'][0]
            if kfold_labels:
                mask = np.isin(ids, kfold_labels)
                print(mask.sum(), 'patients in fold')
                x = x[mask]
                y_all = y_all[mask]
                ids = ids[mask]
            y_2 = np.stack([y_all == 0, y_all == 1], axis=1).astype(int)
            y = normalize(y_2, axis=0)
            X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                                test_size=0.33,
                                                                random_state=42)
        else:
            raise NotImplementedError('Count connectivity matrix by yourself')

        if self.mode == "train":
            x = X_train
            y = y_train
        elif self.mode == "validation":
            x = X_test
            y = y_test
        else:
            x = x
            y = y

        self.X = torch.FloatTensor(np.expand_dims(x, 1).astype(np.float32))
        self.Y = torch.FloatTensor(y.astype(np.float32))

        print(self.mode, self.X.shape, (self.Y.shape))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = [self.X[idx], self.Y[idx]]
        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample


def _get_adjmat(patient: pd.DataFrame, atlas):
    label_map = {l:i for i,l in enumerate(atlas.labels)}
    connectivity_mat = np.zeros((len(label_map), len(label_map)))
    for name1 in label_map.keys():
        for name2 in label_map.keys():
            col_name = f'{name1}-{name2}'
            
            connectivity_mat[label_map[name1]][label_map[name2]] = patient[col_name] if col_name in patient.columns else 0.0
    connectivity_mat = connectivity_mat + np.diag(np.ones(len(connectivity_mat))) + np.triu(connectivity_mat).T
    return connectivity_mat


def _get_norm_adj_mats(conn_matr_df: pd.DataFrame, atlas):
    '''
        Arguments:
            conn_matr_df: dataframe with roi atlas values for each subject
            atlas: atlas for rois
        Returns:
            centrally normalized connectivity matrix [n_patients, len(atlas), len(atlas)]
    '''
    conn_matr_norm = np.stack([_get_adjmat(pd.DataFrame(pat).T, atlas) for i,pat in conn_matr_df.iterrows()])
    return conn_matr_norm


class BrainGNNDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super(BrainGNNDataset, self).__init__(root,transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        data_dir = os.path.join(self.root,'raw')
        onlyfiles = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        onlyfiles.sort()
        return onlyfiles
    @property
    def processed_file_names(self):
        return  'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        # Read data into huge `Data` list.
        self.data, self.slices = read_data(self.raw_dir)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))



datasets = {
    'graph_dataset': GraphDataset,
    'BrainNetCNN'  : BrainNetDataset,
    'BrainGNN'     : BrainGNNDataset,
}


def create_dataset(dataset_name='graph_dataset', save_folder='schiza', **kwargs):
    save_dir = f"{PATHS['dataset_dir']}/{dataset_name}/{save_folder}"
    dataset = datasets[dataset_name](save_dir, **DATASET_PARAMS[dataset_name], **kwargs)
    if dataset_name == 'graph_dataset':
        dataset = dataset.shuffle()
    return dataset


def create_kfold_datasets(dataset_name: str, splits: Dict[int: Dict[str: List[str]]],
                         save_folder='schiza', **kwargs):
    datasets = []
    for i, fold in enumerate(splits):
        train_ids, val_ids = fold['train'], fold['valid']
        save_dir = f"{PATHS['dataset_dir']}/{dataset_name}/{save_folder}_fold{i}"
        train_set = datasets[dataset_name](save_dir, mode='train', kfold_labels=train_ids,
                                           **DATASET_PARAMS[dataset_name], **kwargs)
        val_set = datasets[dataset_name](save_dir, mode='validation', kfold_labels=val_ids,
                                         **DATASET_PARAMS[dataset_name], **kwargs)
        datasets.append((train_set, val_set))
    return datasets
