import logging
import random
from collections import Counter

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

import torch
from torch.utils.data import DataLoader, Dataset


# training/validation dataset generation

def cap_list(ls, n=100, split=.8, oversample=True):
    """Cap list at n.
    If n is larger than list size * .8, oversample until you hit n.
    """
    cap = int(len(ls) * split)
    if cap > n:
        return random.sample(ls, n)

    if oversample:
        pool = random.sample(ls, cap) if cap else list(ls)
        # oversample to
        return random.choices(pool, k=n)

    return random.sample(ls, cap)


def balancedish_training_generator(adata, cell_type_key, n_per_cell_type,
                                   oversample=True, split=.8):
    """Split anndata object into training and validation objects.
    When splitting, equal numbers of cell types will be used for training.
    """
    cell_type_to_idxs = {}
    for cell_id, cell_type in zip(adata.obs.index, adata.obs[cell_type_key]):
        if cell_type not in cell_type_to_idxs:
            cell_type_to_idxs[cell_type] = [cell_id]
        else:
            cell_type_to_idxs[cell_type].append(cell_id)

    cell_type_to_idxs = {k: cap_list(ls, n_per_cell_type, oversample=oversample,
                         split=split)
                         for k, ls in cell_type_to_idxs.items()}

    train_ids = np.asarray([x for ls in cell_type_to_idxs.values() for x in ls])
    train_idxs = np.arange(adata.shape[0])[np.isin(np.asarray(adata.obs.index), train_ids)]
    val_idxs = np.delete(np.arange(adata.shape[0]), train_idxs)

    train_adata = adata[train_idxs, :]
    val_adata = adata[val_idxs, :]

    return train_adata, val_adata


def balance_adata(adata, key):
    """Oversample imbalanced classes so each group (specified by key)
    has the same number of cells per group"""
    n = Counter(adata.obs[key]).most_common()[0][1]
    idxs = []
    for k in sorted(set(adata.obs[key])):
        filtered = adata[adata.obs[key]==k]
        ids = filtered.obs.index.to_list()
        if len(ids) >= n:
            idxs += ids
        else:
            idxs += list(np.random.choice(ids, n, replace=True))
    return adata[idxs]


# adapted from scDCC https://github.com/ttgump/scDCC/blob/master/preprocess.py
def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True,
              var_order=None):

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    # add/reorder vars if needed
    if var_order is not None:
        obs = adata.obs
        a, b = set(var_order), set(adata.var.index.to_list())
        overlap = list(a.intersection(b))
        missing = list(a - set(overlap))
        logging.info(f'{len(overlap)} genes overlap with model after filtering')
        logging.info(f'{len(missing)} genes missing from dataset after filtering')

        new = adata[:, overlap]
        m = anndata.AnnData(X=np.zeros((adata.shape[0], len(missing))), obs=adata.obs)
        m.var.index = missing
        new = anndata.concat((new, m), axis=1)

        adata = new[:, var_order]
        adata.obs = obs

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata


class PollockDataset(Dataset):
    """Pollock Dataset"""
    def __init__(self, adata, label_col='cell_type'):
        self.adata = adata

        if label_col in adata.obs.columns:
            self.cell_types = sorted(set(adata.obs[label_col]))
            self.labels = adata.obs[label_col].to_list()
        else:
            self.cell_types = None
            self.labels = None

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        x = self.adata.X[idx]
        if 'sparse' in str(type(x)).lower():
            x = x.toarray()

        x_raw = self.adata.raw.X[idx]
        if 'sparse' in str(type(x_raw)).lower():
            x_raw = x_raw.toarray()

        x, x_raw = np.squeeze(x), np.squeeze(x_raw)

        sf = self.adata.obs['size_factors'][idx]

        label = self.labels[idx] if self.labels is not None else None
        y = self.cell_types.index(label) if label is not None else None

        return {
            'x': x,
            'x_raw': x_raw,
            'size_factor': sf,
            'y': y,
            'label': label
        }

def get_train_dataloaders(train, val, batch_size=64, label_col='cell_type'):
    """
    Get train and validation dataloaders.

    Arguments
    ---------
    train: str or AnnData
        - AnnData object or filepath of saved AnnData with .h5ad ext to be used for training
    val: str or AnnData
        - AnnData object or filepath of saved AnnData with .h5ad ext to be used for validation
    batch_size: int
        - batch size for dataloaders
    """
    train_adata = sc.read_h5ad(train) if isinstance(train, str) else train
    val_adata = sc.read_h5ad(val) if isinstance(val, str) else val

    train_adata = normalize(train_adata)
    val_adata = normalize(val_adata, var_order=train_adata.var.index.to_list())

    train_ds = PollockDataset(train_adata, label_col=label_col)
    val_ds = PollockDataset(val_adata, label_col=label_col)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl

def get_prediction_dataloader(adata, var_order, batch_size=64):
    adata = sc.read_h5ad(adata) if isinstance(adata, str) else adata

    adata = normalize(adata, var_order=var_order)

    ds = PollockDataset(adata)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    return dl
