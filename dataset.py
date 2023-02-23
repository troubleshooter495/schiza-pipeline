from torch_geometric.data import InMemoryDataset
import torch
import numpy as np

from torch_geometric.utils import from_networkx
from networkx.convert_matrix import from_numpy_matrix
from nilearn.connectome import ConnectivityMeasure

import torch.utils.data.dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from utils.io import read_file_locally
from config import PATHS, DATASET_PARAMS, SEED


class GraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, neighbors=10):
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


class BrainNetCNNDataset(torch.utils.data.Dataset):
    def __init__(self, save_dir: str, load_dir: str, mode: str,
                 transform=False):
        """
        Args:
            directory (string): Path to the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.directory = load_dir
        self.transform = transform
        self.mode = mode

        x = read_file_locally(load_dir)['dti']
        y_all = read_file_locally(load_dir)['label'][0]
        y_2 = np.stack([y_all == 0, y_all == 1], axis=1).astype(int)
        y = normalize(y_2, axis=0)
        X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.33,
                                                            random_state=42)

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


datasets = {
    'graph_dataset': GraphDataset,
    'BrainNetCNN': BrainNetCNNDataset,
}


def create_dataset(dataset_name='graph_dataset', save_folder='schiza', **kwargs):
    save_dir = f"{PATHS['dataset_dir']}/{dataset_name}/{save_folder}"
    dataset = datasets[dataset_name](save_dir, **DATASET_PARAMS[dataset_name], **kwargs)
    if dataset_name == 'graph_dataset':
        dataset = dataset.shuffle()
    return dataset
