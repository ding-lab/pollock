import argparse
import os

import pandas as pd
import numpy as np
import sklearn
import scanpy as sc
import anndata
import shap

from pollock.models.model import load_from_directory

# we have to disable v2 things for shap gradient explainer to work
from tensorflow.compat.v1.keras.backend import get_session
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

parser = argparse.ArgumentParser()

parser.add_argument('explain_anndata_filepath', type=str,
        help='filepath of .h5ad anndata object holding raw expression \
counts of cells whose embeddings are to be explained')

parser.add_argument('background_anndata_filepath', type=str,
        help='filepath of .h5ad anndata object holding raw expression \
counts to be used as background values')

parser.add_argument('module_filepath', type=str,
        help='filepath of pollock module to use for the shap values.')

parser.add_argument('output_filepath', type=str,
        help='filepath to write output .npy array of shap values')

args = parser.parse_args()

def get_shaps(explain_adata, background_adata, pm, n_layer=0):
    """Get shap values for the cells in the given anndata object."""
    model = pm.model.layers[0]

    def map2layer(x, layer):
        feed_dict = dict(zip([model.layers[0].input], [x]))
        return get_session().run(model.layers[layer].input, feed_dict)

    e = shap.GradientExplainer(
        (model.layers[n_layer].input, model.layers[-1].output),
        map2layer(background_adata.X, n_layer))

    shap_values = e.shap_values(map2layer(explain_adata.X, n_layer))

    # this returns in the shape of (n_cells, n_embedding, n_features)
    return np.swapaxes(np.asarray(shap_values[:pm.latent_dim]), 0, 1)

if __name__ == '__main__':
    module_fp = args.module_filepath
    adata_explain = sc.read_h5ad(args.explain_anndata_filepath)
    adata_background = sc.read_h5ad(args.background_anndata_filepath)
    explain_pds, _ = load_from_directory(adata_explain, module_fp)
    background_pds, pm = load_from_directory(adata_background, module_fp)

    shaps = get_shaps(explain_pds.prediction_adata.copy(),
            background_pds.prediction_adata.copy(), pm)

    np.save(args.output_filepath, shaps)
