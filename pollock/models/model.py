import logging
import json
import joblib
import math
import os
import random
import re
import shutil
import time
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
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
import umap

from tensorflow.keras.models import Sequential
import tensorflow as tf

import pollock.models.analysis as pollock_analysis

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

MODEL_PATH = 'model.h5' 
CELL_TYPES_PATH = 'cell_types.npy' 
GENES_PATH = 'genes.npy' 
MODEL_SUMMARY_PATH = 'summary.json' 
STANDARD_SCALER_PATH = 'standard_scaler.pkl' 
RANGE_SCALER_PATH = 'range_scaler.pkl' 
ENCODER_PATH = 'encoder.pkl' 
CLASSIFIER_PATH = 'clf.pkl'

def cap_list(ls, n=100):
    if len(ls) > n:
        return random.sample(ls, n)
    return random.sample(ls, int(len(ls) * .8))

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
    train_idxs = np.arange(adata.shape[0])[np.isin(np.asarray(adata.obs.index), train_ids)]
    val_idxs = np.delete(np.arange(adata.shape[0]), train_idxs)

    train_adata = adata[train_idxs, :]
    val_adata = adata[val_idxs, :]

    return train_adata, val_adata

def get_tf_datasets(train_adata, val_adata, train_buffer=10000, batch_size=64):
    if 'sparse' in str(type(train_adata.X)).lower():
        X_train = train_adata.X.toarray()
    else:
        X_train = train_adata.X

    if 'sparse' in str(type(val_adata.X)).lower():
        X_val = val_adata.X.toarray()
    else:
        X_val = val_adata.X

    train_dataset = tf.data.Dataset.from_tensor_slices(X_train
            ).shuffle(train_buffer).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(X_val
            ).batch(batch_size)

    return train_dataset, val_dataset

def get_tf_prediction_ds(adata, batch_size=1000):
    if 'sparse' in str(type(adata.X)).lower():
        X = adata.X.toarray()
    else:
        X = adata.X

    dataset = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)

    return dataset

def process_from_counts(adata, min_genes=200, min_cells=3, mito_threshold=.2, max_n_genes=None,
        log=True, cpm=True, min_disp=.2, standard_scaler=None, range_scaler=None,
        normalize_samples=True):
    if min_genes is not None:
        logging.info(f'filtering by min genes: {min_genes}')
        sc.pp.filter_cells(adata, min_genes=min_genes)
        logging.info(f'cells remaining after filter: {adata.shape[0]}')
    if min_cells is not None:
        logging.info(f'filtering by min cells: {min_cells}')
        sc.pp.filter_genes(adata, min_cells=min_cells)
        logging.info(f'genes remaining after filter: {adata.shape[1]}')

    if mito_threshold is not None or max_n_genes is not None: 
        logging.info('calculating MT and gene counts')
        mito_genes = adata.var_names.str.startswith('MT-')
        if 'sparse' in str(type(adata.X)).lower():
            adata.obs['percent_mito'] = np.sum(
                adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
            adata.obs['n_counts'] = adata.X.sum(axis=1).A1
        else:
            adata.obs['percent_mito'] = np.sum(
                adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
            # add the total counts per cell as observations-annotation to adata
            adata.obs['n_counts'] = adata.X.sum(axis=1)
    
        if mito_threshold is not None:
            logging.info('filtering by mito threshold')
            adata = adata[adata.obs.percent_mito < mito_threshold, :]
        if max_n_genes is not None:
            logging.info('filtering by n genes threshold')
            adata = adata[adata.obs.n_genes < max_n_genes, :]

    if log:
        logging.info('loging data')
        sc.pp.log1p(adata)
    adata.raw = adata
    
    if min_disp is not None:
        logging.info(f'filtering with dispersion {min_disp}')
        sc.pp.highly_variable_genes(adata, min_mean=None, max_mean=None, min_disp=min_disp)
        remaining = np.count_nonzero(adata.var.highly_variable)
        logging.info(f'remaining after min disp: {remaining}')
        adata = adata[:, adata.var.highly_variable]


    logging.info('scaling data')

    if standard_scaler is None:
        standard_scaler = StandardScaler(with_mean=False, with_std=True)
        adata.X = standard_scaler.fit_transform(adata.X)
    else:
        adata.X = standard_scaler.transform(adata.X)

    if range_scaler is None:
        range_scaler = MaxAbsScaler()
        adata.X = range_scaler.fit_transform(adata.X)
    else:
        ## some minmaxscalers were saved with model and wont work on sparse
        if 'maxabsscaler' not in str(type(range_scaler)).lower():
            range_scaler = MaxAbsScaler()
            range_scaler.fit(adata.X)
            
        adata.X = range_scaler.transform(adata.X)

    return adata, standard_scaler, range_scaler

def filter_adata_genes(adata, genes):
    logging.info('filtering for genes in training set')
    training_genes = set(genes)
    prediction_genes = set(adata.var.index)
    missing = list(training_genes - prediction_genes)
    logging.info(f'{len(missing)} genes in training set are missing from prediction set')

    X = adata.X
    if 'sparse' in str(type(X)).lower():
        X = scipy.sparse.hstack([X, scipy.sparse.coo_matrix((adata.shape[0], len(missing)))])
        X = X.tocsr()
    else:
        X = np.concatenate((X, np.zeros((adata.shape[0], len(missing)))), axis=1)
    var_index = list(adata.var.index) + missing
    new_adata = anndata.AnnData(X=X, obs=adata.obs)
    new_adata.var.index = var_index

    return new_adata[:, genes]

class PollockDataset(object):
    def __init__(self, adata, cell_type_key='ClusterName', n_per_cell_type=500,
            batch_size=64, dataset_type='training', min_genes=10, min_cells=3, mito_threshold=None,
            max_n_genes=None, log=True, cpm=False, min_disp=None, standard_scaler=None,
            range_scaler=None, cell_type_encoder=None, genes=None, cell_types=None):

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
        self.genes = genes

        if dataset_type == 'prediction':
            self.set_prediction_dataset()
        else:
            self.cell_type_key=cell_type_key
            self.cell_types = sorted(set(self.adata.obs[self.cell_type_key]))
            self.cell_type_encoder = OrdinalEncoder(categories=[self.cell_types])
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
        self.adata, self.standard_scaler, self.range_scaler = process_from_counts(self.adata,
                min_genes=self.min_genes, min_cells=self.min_cells, mito_threshold=self.mito_threshold,
                max_n_genes=self.max_n_genes, log=self.log, cpm=self.cpm, min_disp=self.min_disp,
                standard_scaler=self.standard_scaler, range_scaler=self.range_scaler)
        self.genes = np.asarray(self.adata.var.index)

        logging.info(f'creating tf datasets')
        self.train_adata, self.val_adata = balancedish_training_generator(self.adata,
                self.cell_type_key, self.n_per_cell_type)

        self.train_ds, self.val_ds = get_tf_datasets(self.train_adata, self.val_adata,
                train_buffer=10000, batch_size=self.batch_size)

        self.train_cell_ids = np.asarray(self.train_adata.obs.index)
        self.val_cell_ids = np.asarray(self.val_adata.obs.index)

        self.y_train = np.asarray(self.train_adata.obs[self.cell_type_key])
        self.y_train = self.cell_type_encoder.fit_transform(self.y_train.reshape(-1, 1)).flatten()
        self.y_val = np.asarray(self.val_adata.obs[self.cell_type_key])
        self.y_val = self.cell_type_encoder.transform(self.y_val.reshape(-1, 1)).flatten()

    def set_prediction_dataset(self):
        logging.info(f'normalizing counts for prediction')
        self.prediction_adata = filter_adata_genes(self.adata, self.genes)
        self.prediction_adata, _, _ = process_from_counts(self.prediction_adata,
                min_genes=self.min_genes, min_cells=self.min_cells, mito_threshold=self.mito_threshold,
                max_n_genes=self.max_n_genes, log=self.log, cpm=self.cpm, min_disp=self.min_disp,
                standard_scaler=self.standard_scaler, range_scaler=self.range_scaler)
        self.prediction_ds = get_tf_prediction_ds(self.prediction_adata, batch_size=1000)

def load_from_directory(adata, model_filepath, batch_size=64, min_genes_per_cell=200):
##     model = tf.saved_model.load(os.path.join(model_filepath, MODEL_PATH))
    cell_types = np.load(os.path.join(model_filepath, CELL_TYPES_PATH), allow_pickle=True)
    genes = np.load(os.path.join(model_filepath, GENES_PATH), allow_pickle=True)
    summary = json.load(open(os.path.join(model_filepath, MODEL_SUMMARY_PATH)))
    standard_scaler = joblib.load(os.path.join(model_filepath, STANDARD_SCALER_PATH))
    range_scaler = joblib.load(os.path.join(model_filepath, RANGE_SCALER_PATH))
    encoder = joblib.load(os.path.join(model_filepath, ENCODER_PATH))
    clf = joblib.load(os.path.join(model_filepath, CLASSIFIER_PATH))

    prediction_dataset = PollockDataset(adata, batch_size=batch_size, dataset_type='prediction',
            min_genes=min_genes_per_cell, min_cells=None, mito_threshold=None,
            max_n_genes=None, log=True, cpm=True, min_disp=None, standard_scaler=standard_scaler,
            range_scaler=range_scaler, genes=genes, cell_type_encoder=encoder,
            cell_types=cell_types)

    latent_dim = summary['model_parameters']['latent_dim'] if 'model_parameters' in summary else 100
    pollock_model = PollockModel(cell_types, input_shape=len(genes),
            latent_dim=latent_dim,
            model=os.path.join(model_filepath, MODEL_PATH), summary=summary, clf=clf)

    return prediction_dataset, pollock_model

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

class BVAE(tf.keras.Model):
    def __init__(self, latent_dim, input_size):
        super(BVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_size,)),
                tf.keras.layers.Dense(1000, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(1000, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ])

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(1000, activation='relu'),
                tf.keras.layers.Dropout(.2),
                tf.keras.layers.Dense(1000, activation='relu'),
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

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

@tf.function
def compute_loss(model, x, alpha=0.01):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    kl_loss = .5 * tf.reduce_sum(tf.exp(logvar) + tf.square(mean) - 1. - logvar, axis=1)
    reconstruction_loss = .5 * tf.reduce_sum(tf.square((x - x_logit)), axis=1)

    overall_loss = tf.reduce_mean(reconstruction_loss + alpha * kl_loss)
    return overall_loss

## @tf.function
## def compute_apply_gradients(model, x, optimizer, alpha=.01):
##     with tf.GradientTape() as tape:
##         loss = compute_loss(model, x, alpha=alpha)
##     gradients = tape.gradient(loss, model.trainable_variables)
##     optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def get_compute_apply_gradients():
    @tf.function
    def cag(model, x, optimizer, alpha=.01):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, x, alpha=alpha)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return cag
compute_apply_gradients = get_compute_apply_gradients()

def batch_adata(adata, n=1000):
    if 'sparse' in str(type(adata.X)).lower():
        return [adata.X[i:i+n].toarray()
                for i in range(0, adata.shape[0], n)]
    else:
        return [adata.X[i:i+n]
                for i in range(0, adata.shape[0], n)]

class PollockModel(object):
    def __init__(self, class_names, input_shape, model=None, learning_rate=1e-4,
            summary=None, alpha=.1,
            latent_dim=100, clf=None):
        tf.keras.backend.clear_session()
        self.model = BVAE(latent_dim, input_shape)
        if model is not None:
            self.model.load_weights(model)

        if clf is None:
            clf = RandomForestClassifier(n_estimators=100)

        self.class_names = class_names
        self.summary = summary
        self.alpha = alpha
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        self.clf = clf
        self.val_losses = []
        self.train_losses = []
        self.val_accuracies = []
        self.train_accuracies = []
        self.cell_type_train_losses = {c:[] for c in self.class_names}
        self.cell_type_val_losses = {c:[] for c in self.class_names}
        self.cell_type_train_accuracies = {c:[] for c in self.class_names}
        self.cell_type_val_accuracies = {c:[] for c in self.class_names}



    def get_cell_embeddings(self, ds):
        embeddings = None
        for X in ds:
            mean, logvar = self.model.encode(X)
            emb = self.model.reparameterize(mean, logvar).numpy()
            if embeddings is None:
                embeddings = emb
            else:
                embeddings = np.concatenate((embeddings, emb), axis=0)

        return embeddings

    def get_umap_cell_embeddings(self, ds):
        embeddings = self.get_cell_embeddings(ds)
        return umap.UMAP().fit_transform(embeddings)

    def get_cell_type_loss(self, cell_adata):
        ## do training
        cell_ids = np.asarray(
                random.sample(list(cell_adata.obs.index), min(cell_adata.shape[0], 100)))
        if 'sparse' in str(type(cell_adata.X)).lower():
            X = cell_adata[cell_ids].X.toarray()
        else:
            X = cell_adata[cell_ids].X
        loss = tf.keras.metrics.Mean()
        loss(compute_loss(self.model, X, alpha=self.alpha))
        return loss.result().numpy()

    def get_cell_type_accuracy(self, X, y, clf=None):
        X = self.get_cell_embeddings(X)
        if clf is None:
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(X, y)

        probs = clf.predict_proba(X)
        output_classes = np.argmax(probs, axis=1).flatten()
        report = classification_report(y,
                output_classes, labels=list(range(len(self.class_names))),
                target_names=self.class_names, output_dict=True, zero_division=0)
        return clf, report

    def fit(self, pollock_dataset, epochs=10, max_metric_batches=10, metric_epoch_interval=5):
        compute_apply_gradients = get_compute_apply_gradients()
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            for train_x in pollock_dataset.train_ds:
                compute_apply_gradients(self.model, train_x, self.optimizer, alpha=self.alpha)
            end_time = time.time()

            train_loss = tf.keras.metrics.Mean()
            for train_x in pollock_dataset.train_ds:
                train_loss(compute_loss(self.model, train_x, alpha=self.alpha))

            val_loss = tf.keras.metrics.Mean()
            ## only take some of validation dataset to avoid unnecessarilly long model training times
            if max_metric_batches is None:
                for test_x in pollock_dataset.val_ds:
                    val_loss(compute_loss(self.model, test_x, alpha=self.alpha))
            else:
                for test_x in pollock_dataset.val_ds.take(max_metric_batches):
                    val_loss(compute_loss(self.model, test_x, alpha=self.alpha))

            if epoch - 1 % metric_epoch_interval == 0:
    
                ## get losses for each cell type
                ## randomly taking min(100, n_cells) to keep running time down
                for cell_type in self.class_names:
                    ## do training
                    cell_adata = pollock_dataset.train_adata[pollock_dataset.train_adata.obs[pollock_dataset.cell_type_key]==cell_type]
                    self.cell_type_train_losses[cell_type].append(self.get_cell_type_loss(cell_adata))
    
                    ## do validation
                    cell_adata = pollock_dataset.val_adata[pollock_dataset.val_adata.obs[pollock_dataset.cell_type_key]==cell_type]
                    self.cell_type_val_losses[cell_type].append(self.get_cell_type_loss(cell_adata))
    
                ## do accuracies
                adata, _ = balancedish_training_generator(pollock_dataset.train_adata,
                        pollock_dataset.cell_type_key, n_per_cell_type=100)
                y = pollock_dataset.cell_type_encoder.transform(
                        adata.obs[pollock_dataset.cell_type_key].to_numpy().reshape(
                            (-1, 1))).flatten()
                X = batch_adata(adata, n=pollock_dataset.batch_size)
                X, y = X[:max_metric_batches], y[:max_metric_batches*pollock_dataset.batch_size]
                clf, report = self.get_cell_type_accuracy(X, y,
                        clf=None)
                self.train_accuracies.append(report['accuracy'])
                for cell_type in self.class_names: 
                    self.cell_type_train_accuracies[cell_type].append(report[cell_type]['f1-score'])
        
                adata, _ = balancedish_training_generator(pollock_dataset.val_adata,
                        pollock_dataset.cell_type_key, n_per_cell_type=100)
                y = pollock_dataset.cell_type_encoder.transform(
                        adata.obs[pollock_dataset.cell_type_key].to_numpy().reshape(
                            (-1, 1))).flatten()
                X = batch_adata(adata, n=pollock_dataset.batch_size)
                X, y = X[:max_metric_batches], y[:max_metric_batches*pollock_dataset.batch_size]
                _, report = self.get_cell_type_accuracy(X, y,
                        clf=clf)
                self.val_accuracies.append(report['accuracy'])
                for cell_type in self.class_names: 
                    self.cell_type_val_accuracies[cell_type].append(report[cell_type]['f1-score'])

            logging.info(f'epoch: {epoch}, train loss: {train_loss.result()}, \
val loss: {val_loss.result()}') 
            self.train_losses.append(train_loss.result().numpy())
            self.val_losses.append(val_loss.result().numpy())

        train_batches = batch_adata(pollock_dataset.train_adata,
                n=pollock_dataset.batch_size)
        X_train = self.get_cell_embeddings(train_batches)
        self.clf.fit(X_train, pollock_dataset.y_train)

        val_batches = pollock_dataset.val_ds
        X_val = self.get_cell_embeddings(val_batches)

    def predict_pollock_dataset(self, pollock_dataset, labels=False, threshold=0.):
        prediction_batches = pollock_dataset.prediction_ds

        if not labels:
            return self.predict(prediction_batches)

        probs = self.predict(prediction_batches)
        output_classes = np.argmax(probs, axis=1).flatten()
        output_probs = np.max(probs, axis=1).flatten()

        output_labels, output_probs = zip(*
                [(pollock_dataset.cell_type_encoder.categories_[0][c], prob) if prob > threshold else ('unknown', prob)
                for c, prob in zip(output_classes, output_probs)])

        return output_labels, output_probs, probs

    def predict(self, ds):
        X = self.get_cell_embeddings(ds)
        probs = self.clf.predict_proba(X)
        return probs

    def save(self, pollock_training_dataset, filepath, score_train=True,
            score_val=True, metadata=None):
        ## create directory if does not exist
        if not os.path.isdir(filepath):
            os.mkdir(filepath)

        model_fp = os.path.join(filepath, MODEL_PATH)
        self.model.save_weights(model_fp)
        np.save(os.path.join(filepath, CELL_TYPES_PATH),
                np.asarray(pollock_training_dataset.cell_types))
        np.save(os.path.join(filepath, GENES_PATH),
                np.asarray(pollock_training_dataset.genes))
        joblib.dump(pollock_training_dataset.standard_scaler, os.path.join(filepath, STANDARD_SCALER_PATH))
        joblib.dump(pollock_training_dataset.range_scaler, os.path.join(filepath, RANGE_SCALER_PATH))
        joblib.dump(pollock_training_dataset.cell_type_encoder, os.path.join(filepath, ENCODER_PATH))
        joblib.dump(self.clf, os.path.join(filepath, CLASSIFIER_PATH))

        if metadata is not None:
            d = metadata
        else:
            d = {}

        d['history'] = {
                'train_losses': self.train_losses,
                'validation_losses': self.val_losses,
                'train_accuracies': self.train_losses,
                'validation_accuracies': self.val_losses,
                'cell_type_train_losses': self.cell_type_train_losses,
                'cell_type_val_losses': self.cell_type_val_losses,
                'cell_type_train_accuracies': self.cell_type_train_accuracies,
                'cell_type_val_accuracies': self.cell_type_val_accuracies,
                }

        d['model_parameters'] = {
                'alpha': self.alpha,
                'learning_rate': self.lr,
                'latent_dim': self.latent_dim,
                'cell_types': self.class_names
                }

        train_batches = batch_adata(pollock_training_dataset.train_adata, n=1000)
##         val_batches = batch_adata(pollock_training_dataset.val_adata, n=1000)
        val_batches = pollock_training_dataset.val_ds

        if score_train:
            d['training'] = self.generate_report_for_dataset(
                    train_batches,
                    pollock_training_dataset.y_train)

        if score_val:
            d['validation'] = self.generate_report_for_dataset(
                    val_batches,
                    pollock_training_dataset.y_val)

        self.summary = d

        json.dump(d, open(os.path.join(filepath, MODEL_SUMMARY_PATH), 'w'),
                cls=NumpyEncoder)

    def generate_report_for_dataset(self, ds, y):
        probs = self.predict(ds)
        predictions = np.argmax(probs, axis=1).flatten()
        predicted_labels = [self.class_names[i] for i in predictions]
        groundtruth = np.asarray([int(i) for i in y])
        groundtruth_labels = [self.class_names[i] for i in groundtruth]

        report = classification_report(groundtruth,
                predictions, target_names=self.class_names, output_dict=True)

        c_df = pollock_analysis.get_confusion_matrix(predictions,
                groundtruth, self.class_names, show=False)

        d = {
            'metrics': report,
            'probabilities': probs,
            'prediction_labels': predicted_labels,
            'groundtruth_labels': groundtruth_labels,
            'confusion_matrix': c_df.values,
            }
        return d
