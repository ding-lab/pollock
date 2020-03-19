import logging
import json
import joblib
import math
import os
import random
import re
import shutil
import uuid
from collections import Counter, Iterable

import anndata
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import scanpy as sc
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler

from tensorflow.keras.models import Sequential
import tensorflow as tf

import pollock.preprocessing.preprocessing as pollock_pp
import pollock.models.analysis as pollock_analysis

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

MODEL_PATH = 'model.h5' 
CELL_TYPES_PATH = 'cell_types.npy' 
MODEL_SUMMARY_PATH = 'summary.json' 

def cap_list(ls, n=100):
    if len(ls) > n:
        return random.sample(ls, n)
    return ls

## def balanced_adata_filter(adata, cell_type_key, n=1000):
##     cell_type_to_idxs = {}
##     for cell_id, cell_type in zip(adata.obs.index, adata.obs[cell_type_key]):
##         if cell_type not in cell_type_to_idxs:
##             cell_type_to_idxs[cell_type] = [cell_id]
##         else:
##             cell_type_to_idxs[cell_type].append(cell_id)
## 
##     
##     cell_type_to_idxs = {k:cap_list(ls, n=n)
##                          for k, ls in cell_type_to_idxs.items()}
##     
##     idxs = np.asarray([x for ls in cell_type_to_idxs.values() for x in ls])
##     return adata[idxs]

def balancedish_training_generator(adata, cell_type_key, n_per_cell_type):
    cell_type_to_idxs = {}
    for cell_id, cell_type in zip(adata.obs.index, adata.obs[cell_type_key]):
        if cell_type not in cell_type_to_idxs:
            cell_type_to_idxs[cell_type] = [cell_id]
        else:
            cell_type_to_idxs[cell_type].append(cell_id)
    
    cell_type_to_idxs = {k:cap_list(ls, n_per_cell_type)
                         for k, ls in cell_type_to_idxs.items()}
    
    train_ids = np.asarray([x for ls in cell_type_to_idxs.values() for x in ls])
    val_ids = np.delete(np.asarray(adata.obs.index), train_ids)

    train_adata = adata[train_ids, :]
    val_adata = adata[val_ids, :]

    return train_adata, val_adata

def get_tf_datasets(train_adata, val_adata, train_buffer=10000, batch_size=64):
    if 'sparse' in str(type(train_adata.X)):
        X_train = train_adata.X.toarray()
    else:
        X_train = train_adata.X

    if 'sparse' in str(type(val_adata.X)):
        X_val = val_adata.X.toarray()
    else:
        X_val = val_adata.X

    train_dataset = tf.data.Dataset.from_tensor_slices(X_train
            ).shuffle(train_buffer).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(X_val
            ).batch(batch_size)

    return train_dataset, val_dataset

def get_tf_prediction_ds(adata, batch_size=1000):
    if 'sparse' in str(type(adata.X)):
        X = adata.X.toarray()
    else:
        X = adata.X

    dataset = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)

    return dataset

def process_from_counts(adata, min_genes=200, min_cells=3, mito_threshold=.2, max_n_genes=None,
        log=True, cpm=True, min_disp=.2, standard_scaler=None, range_scaler=None):
    if min_genes is not None:
        sc.pp.filter_cells(adata, min_genes=min_genes)
    if min_cells is not None:
        sc.pp.filter_genes(adata, min_cells=min_cells)
    
    if mito_threshold is not None or max_n_genes is not None: 
        mito_genes = tumor_adata.var_names.str.startswith('MT-')
        if 'sparse' in str(type(adata.X)):
            adata.obs['percent_mito'] = np.sum(
                adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
            adata.obs['n_counts'] = adata.X.sum(axis=1).A1
        else
            adata.obs['percent_mito'] = np.sum(
                adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
            # add the total counts per cell as observations-annotation to adata
            adata.obs['n_counts'] = adata.X.sum(axis=1)
    
        if mito_threshold is not None:
            adata = adata[adata.obs.percent_mito < mito_threshold, :]
        if max_n_genes is not None:
            adata = adata[adata.obs.n_genes < max_n_genes, :]

    if cpm:
        sc.pp.normalize_total(adata, target_sum=1e6)
    if log:
        sc.pp.log1p(adata)
    adata.raw = adata
    
    if min_disp is not None:
        sc.pp.highly_variable_genes(adata, min_disp=min_disp)
        remaining = np.count_nonzero(adata.var.highly_variable)
        logging.info('remaining after min disp: {remaining}')
        adata = adata[:, adata.var.highly_variable]

    if standard_scaler is None:
        standard_scaler = StandardScaler(with_mean=False, with_std=True)
        adata.X = standard_scaler.fit_transform(adata.X)
    else:
        adata.X = standadrd_scaler.transofmr(adata.X)

    if range_scaler is None:
        range_scaler = MinMaxScaler()
        adata.X = range_scaler.fit_transform(adata.X)
    else:
        adata.X = range_scaler.transform(adata.X)

    return adata

class PollockDataset(object):
    def __init__(self, adata, cell_type_key='ClusterName', n_per_cell_type=500,
            batch_size=64, dataset_type='training', min_genes=200, min_cells=3, mito_threshold=.2,
            max_n_genes=None, log=True, cpm=True, min_disp=.2, standard_scaler=None,
            range_scaler=None, cell_type_encoder=None):

        self.adata = adata
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.cell_types = cell_types
        self.min_genes = min_genes
        self.min_cells = min_cells
        self.mito_threshold = mito_threshold
        self.max_n_genes = max_n_genes
        self.log = log
        self.cpm = cpm
        self.min_disp = min_disp
        self.standard_scaler = standard_scaler
        self.range_scaler = range_scaler
        self.cell_type_encoder = cell_type_encoder

        if dataset_type == 'prediction':
            self.set_prediction_dataset()
        else:
            self.cell_type_key=cell_type_key
            self.cell_types = sorted(set(self.adata.obs[self.cell_type_key]))
            self.cell_type_encoder = OrdinalEncoder(categories=self.cell_types)
            self.n_per_cell_type=n_per_cell_type
            self.train_adata = None
            self.val_adata = None
            self.train_ds = None
            self.val_ds = None
            self.train_cell_ids = None
            self.val_cell_ids = None

            self.set_training_datasets()

    def set_training_datasets(self):
        """"""
        logging.info(f'normalizing counts for model training')
        self.adata = process_from_counts(self.adata,
                min_genes=self.min_genes, min_cells=self.min_cells, mito_threshold=self.mito_threshold,
                max_n_genes=self.max_n_genes, log=self.log, cpm=self.cpm, min_disp=self.min_disp,
                standard_scaler=self.standard_scaler, range_scaler=self.range_scaler)

        logging.info(f'creating datasets')
        self.train_adata, self.val_data = balancedish_training_generator(self.adata,
                self.cell_type_key, self.n_per_cell_type)

        self.train_ds, self.val_ds = get_training_tf_dataset(train_adata, val_adata,
                train_buffer=10000, batch_size=self.batch_size)

        self.train_cell_ids = np.asarray(self.train_adata.obs.index)
        self.val_cell_ids = np.asarray(self.val_adata.obs.index)

        self.y_train = np.asarray(self.train_adata.obs[self.cell_type_key])
        self.y_train = self.cell_type_encoder.transform(self.y_train.reshape(-1, 1)).flatten()
        self.y_val = np.asarray(self.val_adata.obs[self.cell_type_key])
        self.y_val = self.cell_type_encoder.transform(self.y_val.reshape(-1, 1)).flatten()

    def set_prediction_dataset(self):
        logging.info(f'normalizing counts for model training')
        self.adata = process_from_counts(self.adata,
                min_genes=self.min_genes, min_cells=self.min_cells, mito_threshold=self.mito_threshold,
                max_n_genes=self.max_n_genes, log=self.log, cpm=self.cpm, min_disp=self.min_disp,
                standard_scaler=self.standard_scaler, range_scaler=self.range_scaler)
        self.prediction_ds = get_tf_prediction_ds(self.adata, batch_size=1000)

def load_from_directory(adata, model_filepath, batch_size=64):
    model = tf.keras.models.load_model(os.path.join(model_filepath, MODEL_PATH))
    cell_types = np.load(os.path.join(model_filepath, CELL_TYPES_PATH), allow_pickle=True)
    genes = np.load(os.path.join(model_filepath, GENES_PATH), allow_pickle=True)
    summary = json.load(open(os.path.join(model_filepath, MODEL_SUMMARY_PATH)))
    standard_scaler = joblib.load(os.path.join(model_filepath, STANDARD_SCALER_PATH))
    range_scaler = joblib.load(os.path.join(model_filepath, RANGE_SCALER_PATH))

    prediction_dataset = PollockDataset(adata, batch_size=batch_size, dataset_type='prediction',
            min_genes=None, min_cells=None, mito_threshold=None,
            max_n_genes=None, log=True, cpm=True, min_disp=None, standard_scaler=standard_scaler,
            range_scaler=range_scaler, genes=genes)

    pollock_model = PollockModel(cell_types, input_shape=len(genes), img_width=gene_template.shape[1],
            img_height=gene_template.shape[0], model=model, summary=summary)

    return prediction_dataset, pollock_model

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class BVAE(tf.keras.Model):
    def __init__(self, latent_dim, input_size):
        super(BVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_size,)),
                tf.keras.layers.Dense(800, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(800, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ])

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(800, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(800, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(input_size),
            ])

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)
    
    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
    
        return logits

optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

@tf.function
def compute_loss(model, x, alpha=0.00005):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    kl_loss = .5 * tf.reduce_sum(tf.exp(logvar) + tf.square(mean) - 1. - logvar, axis=1)
    reconstruction_loss = .5 * tf.reduce_sum(tf.square((x - x_logit)), axis=1)

    overall_loss = tf.reduce_mean(reconstruction_loss + alpha * kl_loss)
    return overall_loss

@tf.function
def compute_apply_gradients(model, x, optimizer, alpha=.00005):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, alpha=alpha)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

class PollockModel(object):
    def __init__(self, class_names, input_shape, model=None, learning_rate=1e-4, summary=None, alpha=.1,
            latent_dim=100, clf=RandomForestClassifier(), encoder=None):
        if model is None:
            model = BVAE(latent_dim, input_shape)
        else:
            self.model = model

        self.class_names = class_names
        self.summary = summary
        self.alpha = alpha
        self.lr = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        self.clf = clf
        self.encoder = encoder
        if self.encoder is None:
            self.encoder = OrdinalEncoder(categories=class_names)

    def get_cell_embeddings(self, ds):
        mean, logvar = self.model.encode(ds)
        return self.model.reparameterize(mean, logvar).numpy()

    def fit(self, pollock_dataset, epochs=10):
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            for train_x in pollock_dataset.train_ds:
                compute_apply_gradients(self.model, train_x, self.optimizer, alpha=self.alpha)
            end_time = time.time()

            loss = tf.keras.metrics.Mean()
            for test_x in pollock_dataset.val_ds:
                loss(compute_loss(self.model, test_x, alpha=self.alpha))

            logging.info(f'epoch: {epoch}, val loss: {loss.result()}') 

        X_train = self.get_cell_embeddings(pollock_dataset.train_ds)

        self.clf.fit(X_train, pollock_dataset.y_train)

    def predict_pollock_dataset(self, pollock_dataset, labels=False, threshold=0.):
        if not labels:
            return self.predict(pollock_dataset.prediction_ds)

        probs = self.model.predict(pollock_dataset.prediction_ds)
        output_classes = np.argmax(probs, axis=1).flatten()
        output_probs = np.max(probs, axis=1).flatten()

        filtered_output_labels, filtered_output_probs = zip(*[(pollock_dataset.cell_types[c], prob)
                for c, prob in zip(output_classes, output_probs)
                if prob > threshold])

        return filtered_output_labels, filtered_output_probs

    def predict(self, ds):
        X = self.get_cell_embeddings(ds)
        self.clf.predict_proba(X)

    def save(self, pollock_training_dataset, filepath, X_val=None, y_val=None,
            X_train=None, y_train=None, metadata=None):
        ## create directory if does not exist
        if not os.path.isdir(filepath):
            os.mkdir(filepath)

        model_fp = os.path.join(filepath, MODEL_PATH)
        self.model.save(model_fp)
        np.save(os.path.join(filepath, GENE_TEMPLATE_PATH),
                pollock_training_dataset.gene_template)
        np.save(os.path.join(filepath, CELL_TYPE_TEMPLATE_PATH),
                pollock_training_dataset.cell_type_template)
        np.save(os.path.join(filepath, CELL_TYPES_PATH),
                np.asarray(pollock_training_dataset.cell_types))
        np.save(os.path.join(filepath, PCA_GENES_PATH),
                np.asarray(pollock_training_dataset.pca_genes))
        joblib.dump(pollock_training_dataset.pca, os.path.join(filepath, PCA_PATH))

        if metadata is not None:
            d = metadata

        d['history'] = {k:[float(x) for x in v]
                for k, v in self.history.history.items()}

        if X_val is not None and y_val is not None:
            d['validation'] = self.generate_report_for_dataset(X_val, y_val)

        if X_train is not None and y_train is not None:
            d['training'] = self.generate_report_for_dataset(X_train, y_train)

        self.summary = d

        json.dump(d, open(os.path.join(filepath, MODEL_SUMMARY_PATH), 'w'),
                cls=NumpyEncoder)

    def generate_report_for_dataset(self, X, y):
        probs = self.model.predict(X)
        predictions = np.argmax(probs, axis=1).flatten()
        predicted_labels = [self.class_names[i] for i in predictions]

        cell_type_to_index = {v:k for k, v in enumerate(self.class_names)}
        groundtruth = [cell_type_to_index[cell_type]
                for cell_type in y]

        report = classification_report(groundtruth, predictions,
               target_names=self.class_names, output_dict=True)

        c_df = pollock_analysis.get_confusion_matrix(predictions,
                groundtruth, self.class_names,  show=False)


        d = {
            'metrics': report,
            'probabilities': probs,
            'prediction_labels': predicted_labels,
            'groundtruth_labels': y,
            'confusion_matrix': c_df.values,
            }
        return d
