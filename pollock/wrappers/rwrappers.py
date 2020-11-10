import os
import logging

import anndata
import pandas as pd
import numpy as np
import scanpy as sc

from pollock.models.model import predict_from_anndata, PollockDataset, PollockModel
from pollock.models.explain import explain_predictions

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
    logging.info('predicting from dataframe')
    adata = anndata.AnnData(X=df.values.transpose())
    adata.obs.index = df.columns
    adata.var.index = df.index.to_list()

    return predict_from_anndata(adata, module_filepath)

def fit_from_dataframe(df, labels, output_filepath, n_per_cell_type=500,
        alpha=.0001, latent_dim=100, epochs=25):
    """fit model from a dataframe.
    """
    adata = anndata.AnnData(X=df.values.transpose())
    adata.obs.index = df.columns
    adata.obs['cell_type'] = labels
    adata.var.index = df.index.to_list()
    logging.info('fitting from dataframe')

    pds = PollockDataset(adata.copy(), cell_type_key='cell_type',
            n_per_cell_type=int(n_per_cell_type),
            batch_size=64, dataset_type='training')

    pm = PollockModel(pds.cell_types, pds.train_adata.shape[1],
            alpha=alpha, latent_dim=int(latent_dim))

    pm.fit(pds, epochs=int(epochs))

    pm.save(pds, output_filepath)

def explain_from_dataframe(explain, background, labels, module_fp,
        background_sample_size=100):
    """Explain predictions from dataframe"""
    explain_adata = anndata.AnnData(X=explain.values.transpose())
    explain_adata.obs.index = explain.columns
    explain_adata.obs['cell_type'] = labels
    explain_adata.var.index = explain.index.to_list()

    background_adata = anndata.AnnData(X=background.values.transpose())
    background_adata.obs.index = background.columns
    background_adata.var.index = background.index.to_list()

    df = explain_predictions(explain_adata, background_adata, module_fp,
            prediction_key='cell_type', n_background_cells=background_sample_size)
    df.index.name = 'cell_id'
    return df
