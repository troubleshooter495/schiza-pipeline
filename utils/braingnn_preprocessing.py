from os import listdir
import os.path as osp
import json
import multiprocessing
from functools import partial
import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.model_selection import KFold
from torch_geometric.utils import remove_self_loops
import networkx as nx
import pandas as pd
import os
from networkx.convert_matrix import from_numpy_matrix
from model.braingnn.imports.gdc import GDC
from torch_sparse import coalesce


def train_val_test_split(kfold = 5, fold = 0):
    n_sub = 96
    id = list(range(n_sub))


    import random
    random.seed(123)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, random_state=123,shuffle = True)
    kf2 = KFold(n_splits=kfold-1, shuffle=True, random_state = 666)


    test_index = list()
    train_index = list()
    val_index = list()

    for tr,te in kf.split(np.array(id)):
        test_index.append(te)
        tr_id, val_id = list(kf2.split(tr))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]

    return train_id,val_id,test_id


def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if data.pos is not None:
        slices['pos'] = node_slice

    return data, slices



def read_data(data_dir, json_path):
    onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
    with open(json_path) as f:
        d = json.load(f)["0"]['train']

    onlyfiles = [file for file in onlyfiles if file[4:-4] in d ]
    onlyfiles.sort()
    batch = []
    pseudo = []
    y_list = []
    edge_att_list, edge_index_list,att_list = [], [], []

    # parallar computing
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    #pool =  MyPool(processes = cores)
    func = partial(read_sigle_data, data_dir)

    import timeit

    start = timeit.default_timer()

    res = pool.map(func, onlyfiles)

    pool.close()
    pool.join()

    stop = timeit.default_timer()

    print('Time: ', stop - start)



    for j in range(len(res)):
        edge_att_list.append(res[j][0])
        edge_index_list.append(res[j][1]+j*res[j][4])
        att_list.append(res[j][2])
        y_list.append(res[j][3])
        batch.append([j]*res[j][4])
        pseudo.append(np.diag(np.ones(res[j][4])))

    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    y_arr = np.stack(y_list)
    edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr).long()  # classification
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos = pseudo_torch )

    data, slices = split(data, batch_torch)

    return data, slices

def get_label(subj_id_col='Subjectid', target_col='Dx'):
    target = pd.read_csv(os.path.join('/home/druzhinina/gnn/BrainGNN_Pytorch/data/cobra_meta_data.tsv'), sep='\t')
    target = target[[subj_id_col, target_col]]

    # check that there are no different labels assigned to the same ID
    max_labels_per_id = target.groupby(subj_id_col)[target_col].nunique().max()
    assert max_labels_per_id == 1, 'Diffrent targets assigned to the same id!'

    # remove duplicates by subj_id
    target.drop_duplicates(subset=[subj_id_col], inplace=True)
    # set subj_id as index
    target.set_index(subj_id_col, inplace=True)

    # leave only Schizo and Control
    target = target[target[target_col].isin(('No_Known_Disorder', 'Schizophrenia_Strict'))].copy()

    # label encoding
    label2idx: dict[str, int] = {x: i for i, x in enumerate(target[target_col].unique())}
    idx2label: dict[int, str] = {i: x for x, i in label2idx.items()}
    target[target_col] = target[target_col].map(label2idx)
    return target

def read_sigle_data(data_dir,filename,use_gdc =False):
    temp = pd.read_csv(osp.join(data_dir, filename))
    # temp = dd.io.load(osp.join(data_dir, filename))

    # read edge and edge attribute
    # pcorr = np.abs(temp['pcorr'][()])
    # num_nodes = pcorr.shape[0]
    #     G = from_numpy_matrix(pcorr)
    temp = temp.iloc[:,1:].values
    corr = np.abs(temp)
    num_nodes = corr.shape[0]
    G = from_numpy_matrix(corr)
    A = nx.to_scipy_sparse_matrix(G)
    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = corr[adj.row[i], adj.col[i]]
    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)
    att = temp
    label = get_label()
    label = label.loc[filename[4:-4]].values

    att_torch = torch.from_numpy(att).float()
    y_torch = torch.from_numpy(label).long()  # classification

    data = Data(x=att_torch, edge_index=edge_index.long(), y=y_torch, edge_attr=edge_att)
    if use_gdc:
        '''
        Implementation of https://papers.nips.cc/paper/2019/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html
        '''
        data.edge_attr = data.edge_attr.squeeze()
        gdc = GDC(self_loop_weight=1, normalization_in='sym',
                  normalization_out='col',
                  diffusion_kwargs=dict(method='ppr', alpha=0.2),
                  sparsification_kwargs=dict(method='topk', k=20,
                                             dim=0), exact=True)
        data = gdc(data)
        return data.edge_attr.data.numpy(),data.edge_index.data.numpy(),data.x.data.numpy(),data.y.data.item(),num_nodes

    else:
        return edge_att.data.numpy(),edge_index.data.numpy(),att,label,num_nodes
