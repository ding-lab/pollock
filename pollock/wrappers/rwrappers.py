import os

import anndata
import pandas as pd
import numpy as np
import scanpy as sc

from pollock.models.model import load_from_directory, PollockDataset, PollockModel

def get_probability_df(labels, label_probs, probs, pds):
    """Return dataframe with labels and probabliities"""
    df = pd.DataFrame.from_dict({
        'cell_id': list(pds.prediction_adata.obs.index),
        'predicted_cell_type': labels,
        'predicted_cell_type_probability': label_probs})
    df = df.set_index('cell_id')

    ## add cell probabilities
    prob_df = pd.DataFrame(data=probs)
    prob_df.index = list(pds.prediction_adata.obs.index)
    prob_df.columns = [f'probability_{c}' for c in pds.cell_type_encoder.categories_[0]]
    prob_df.columns = [c.replace(' ', '_') for c in prob_df.columns]

    return pd.concat((df, prob_df), axis=1)

def predict_from_dataframe(df, module_filepath):
    """Perdict cell types from a dataframe.

    rows are cell ids, cols are gene names
    """
    adata = anndata.AnnData(X=df.values.transpose())
    adata.obs.index = df.columns
    adata.var.index = df.index.to_list()

    loaded_pds, loaded_pm = load_from_directory(adata, module_filepath) 
    labels, label_probs, probs = loaded_pm.predict_pollock_dataset(loaded_pds, labels=True)

    return get_probability_df(labels, label_probs, probs, loaded_pds)

def fit_from_dataframe(df, labels, output_filepath, n_per_cell_type=500,
        alpha=.0001, latent_dim=100, epochs=25):
    """fit model from a dataframe.
    """
    adata = anndata.AnnData(X=df.values.transpose())
    adata.obs.index = df.columns
    adata.obs['cell_type'] = labels
    adata.var.index = df.index.to_list()

    pds = PollockDataset(adata.copy(), cell_type_key='cell_type',
            n_per_cell_type=n_per_cell_type,
            batch_size=64, dataset_type='training', log=True)

    pm = PollockModel(pds.cell_types, pds.train_adata.shape[1],
            alpha=alpha, latent_dim=latent_dim)

    pm.fit(pds, epochs=epochs)

    pm.save(pds, output_filepath)
