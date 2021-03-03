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

def cap_list(ls, n=100, split=.8, oversample=True):
    """Cap list at n.

    If n is larger than list size * .8, oversample until you hit n.
    """
    cap = int(len(ls) * split)
    if cap > n:
        return random.sample(ls, n)

    if oversample:
        pool = random.sample(ls, cap) if cap else list(ls)
        ## oversample to 
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
    
    cell_type_to_idxs = {k:cap_list(ls, n_per_cell_type, oversample=oversample,
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


def get_tf_datasets(train_adata, val_adata, train_buffer=10000, batch_size=32):
    """Return tf datasets for traiing and validation data"""
    if 'sparse' in str(type(train_adata.X)).lower():
        X_train = train_adata.X.toarray()
    else:
        X_train = train_adata.X.astype(np.float32)

    if 'sparse' in str(type(val_adata.X)).lower():
        X_val = val_adata.X.toarray()
    else:
        X_val = val_adata.X.astype(np.float32)

    train_dataset = tf.data.Dataset.from_tensor_slices(X_train
            ).shuffle(train_buffer).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(X_val
            ).batch(batch_size)

    return train_dataset, val_dataset

def get_tf_prediction_ds(adata, batch_size=1000):
    """Get tf dataset for prediction inputs"""
    if 'sparse' in str(type(adata.X)).lower():
        X = adata.X.toarray()
    else:
        X = adata.X.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)

    return dataset

def process_from_counts(adata, standard_scaler=None, range_scaler=None):
    """Process anndata object for input into BVAE.

    adata is assumed to have unnormalized raw counts in adata.X attribute

    counts are scaled by standard dev and mean centered. They are then
    range scaled to between zero and one
    """
    if standard_scaler is None:
        standard_scaler = StandardScaler(with_mean=False, with_std=True)
        adata.X = standard_scaler.fit_transform(adata.X).astype(np.float32)
    else:
        adata.X = standard_scaler.transform(adata.X).astype(np.float32)

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
    """Removes genes from adata not found in train genes list"""
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
    def __init__(self, adata, cell_type_key='cell_type', n_per_cell_type=500,
            val_max_n_per_cell_type=1000, oversample=True,
            batch_size=64, dataset_type='training', standard_scaler=None, validation_key=None,
            range_scaler=None, cell_type_encoder=None, genes=None, cell_types=None):
        """Dataset for use with Pollock model

        Pollock dataset can be used to fit a Pollock model.

        For usage examples see https://github.com/ding-lab/pollock/tree/master/examples

        Arguments
        ---------
        adata : anndata.AnnData
            AnnData object that holds expression counts for a given dataset.
        cell_type_key : str
            column in adata.obs that holds cell annotations that will be used
            for training
        n_per_cell_type : int
            number of cells of each cell type (in the cell_type_key column)
            that will be used in training dataset. The total number of cells
            in the training dataset will be n_per_cell_type * number_of_possible_cell_types
        val_max_n_per_cell_type : int
            maximum number of cells of each cell type (in the cell_type_key column)
            that will be used in the validation dataset during training. The lower
            the number, the faster training will be. This also helps with memory
            issues for extremly large datasets
        oversample : bool
            Whether to oversample rarer cell types to ensure that n_per_cell_type cells are
            present in the training dataset if the total number of cells is less than
            n_per_cell_type
        batch_size : int
            minibatch size for BVAE
        dataset_type : str
            Can be either 'training' or 'prediction'. Specifies whether the pollock
            dataset is to be used for training or prediction.
        standard_scaler : sklearn.preprocessing.StandardScaler, None
            Standard scaler to be used to scale count data. Only relavent
            if creating prediction pollock dataset.
        range_scaler : sklearn.preprocessing.MinMaxScaler, None
            Range scaler to be used to scale count data between zero and one.
            Only relavent if creating prediction pollock dataset.
        validation_key : str, None
            If creating a training dataset and you want to specify which
            cells should be in the training set and which should be in 
            the validation set, then you can include a boolean column
            in adata.obs that is True for cells that are to be in the
            validation set, and false for cells that should be in the
            training set. validation_key is the name of this column.
            If validation key is None, then the training and validation
            sets are automatically generated.
        cell_type_encoder : sklearn.preprocessing.LabelEncoder, None
            encoder storing mapping from cell type labels to integers.
            Only relavent if creating prediction pollock dataset.
        genes : Collection, None
            gene order expected for model inputs. Only releavent if
            creating prediction pollock dataset.
        cell_types: Collection, None
            possible cell types that can be predicted by the pollock
            model. Only relavent if creating prediction pollock datset.

        Returns
        -------
        pollock.model.models.PollockDataset
        """

        self.adata = adata
        # check for sparse matrix
        # needs to be made nonsparse for normalization to work
        if 'sparse' in str(type(self.adata.X)):
            self.adata.X = self.adata.X.astype(np.int64).toarray()
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.cell_types = cell_types
        self.standard_scaler = standard_scaler
        self.range_scaler = range_scaler
        self.cell_type_encoder = cell_type_encoder
        self.genes = genes

        if dataset_type == 'prediction':
            self.set_prediction_dataset()
        else:
            self.cell_type_key = cell_type_key
            self.cell_types = None
            self.oversample = oversample
            self.cell_type_encoder = None
            self.n_per_cell_type = n_per_cell_type
            self.val_max_n_per_cell_type = val_max_n_per_cell_type
            self.val_key = validation_key
            self.train_adata = None
            self.val_adata = None
            self.train_ds = None
            self.val_ds = None
            self.train_cell_ids = None
            self.val_cell_ids = None

            self.set_training_datasets()

    def set_training_datasets(self):
        """Process datasets for training"""
        self.adata, self.standard_scaler, self.range_scaler = process_from_counts(self.adata,
                standard_scaler=self.standard_scaler, range_scaler=self.range_scaler)
        self.genes = np.asarray(self.adata.var.index)
        logging.info(f'input dataset shape: {self.adata.shape}')

        logging.info(f'possible cell types: {sorted(set(self.adata.obs[self.cell_type_key]))}')

        if self.val_key is None:
            self.train_adata, val_adata = balancedish_training_generator(self.adata,
                self.cell_type_key, self.n_per_cell_type, oversample=self.oversample)
            # resample validation data to cut down on overall size
            self.val_adata, _ = balancedish_training_generator(val_adata,
                self.cell_type_key, self.val_max_n_per_cell_type, oversample=False,
                split=1.)
        else:
            logging.info('using validation key')
            self.train_adata = self.adata[~self.adata.obs[self.val_key]]
            self.val_adata = self.adata[self.adata.obs[self.val_key]]

        self.cell_types = sorted(set(self.train_adata.obs[self.cell_type_key]))
        self.cell_type_encoder = OrdinalEncoder(categories=[self.cell_types])

        self.train_ds, self.val_ds = get_tf_datasets(self.train_adata, self.val_adata,
            train_buffer=10000, batch_size=self.batch_size)

        self.train_cell_ids = np.asarray(self.train_adata.obs.index)
        self.val_cell_ids = np.asarray(self.val_adata.obs.index)

        self.y_train = np.asarray(self.train_adata.obs[self.cell_type_key])
        self.y_train = self.cell_type_encoder.fit_transform(self.y_train.reshape(-1, 1)).flatten()
        # just artificially assign a zero label for cells in validation that arent in training
        self.y_val = np.asarray([x if x in self.cell_types else self.cell_types[0] 
                for x in self.val_adata.obs[self.cell_type_key]])
        self.y_val = self.cell_type_encoder.transform(self.y_val.reshape(-1, 1)).flatten()

    def set_prediction_dataset(self):
        """process dataset for prediction"""
        self.prediction_adata = filter_adata_genes(self.adata, self.genes)
        self.prediction_adata, _, _ = process_from_counts(self.prediction_adata,
                standard_scaler=self.standard_scaler, range_scaler=self.range_scaler)
        self.prediction_ds = get_tf_prediction_ds(self.prediction_adata, batch_size=1000)


def predict_from_anndata(adata, model_filepath, adata_batch_size=10000):
    """Convenience method to predict cell types from
    AnnData object in one shot.

    This method will optionally also batch the adata object
    to avoid memory overflow issues.

    Arguments
    ---------
    adata : anndata.AnnData
        AnnData object holding single cell data. Expression counts
        must be unnormalized
    model_filepath : str
        Path to saved Pollock module to use for prediction
    adata_batch_size : int
        will batch the adata objects into groups of
        adata_batch_size cells to avoid memory overflow issues

    Returns
    -------
    pandas.DataFrame
        dataframe with the following columns: cell id, cell type prediction,
        probability of cell type prediction.
        Additionally there are also columns for the probabilities associated
        with each possible cell label.
    """
    _, pm = load_from_directory(adata[:100].copy(), model_filepath)

    predictions = None
    n, c = adata.shape[0], 0
    for i in range(0, n, adata_batch_size):
        c += 1
        logging.info(f'starting batch {c} of {int(n/adata_batch_size) + 1}')
        tiny = adata[i:i + adata_batch_size]
        (cell_types, genes, summary, standard_scaler,
            range_scaler, encoder, clf) = parse_module_directory(model_filepath)
        pds = PollockDataset(tiny, batch_size=64, dataset_type='prediction',
                standard_scaler=standard_scaler, range_scaler=range_scaler,
                genes=genes, cell_type_encoder=encoder, cell_types=cell_types)
        labels, probs, cell_type_probs = pm.predict_pollock_dataset(pds, labels=True)
        df = pd.DataFrame.from_dict({
            'predicted_cell_type': labels,
            'cell_type_probability': probs,
        })
        df.index = tiny.obs.index
        df = pd.concat((df, pd.DataFrame(data=cell_type_probs, index=tiny.obs.index,
            columns=[f'probability_{c}' for c in pds.cell_types])), axis=1)
        if predictions is None:
            predictions = df
        else:
            predictions = pd.concat((predictions, df))
    predictions.index.name = 'cell_id'
    return predictions


def embed_from_anndata(adata, model_filepath,
            adata_batch_size=10000):
    """Convenience method to get BVAE cell embeddings from
    AnnData object in one shot.

    This method will optionally also batch the adata object
    to avoid memory overflow issues.

    Arguments
    ---------
    adata : anndata.AnnData
        AnnData object holding single cell data. Expression counts
        must be unnormalized
    model_filepath : str
        Path to saved Pollock module to use for prediction
    adata_batch_size : int
        will batch the adata objects into groups of
        adata_batch_size cells to avoid memory overflow issues

    Returns
    -------
    pandas.DataFrame
        dataframe where each column corresponds to a position in the
        BVAE latent embedding and each row is a cell.
    """
    _, pm = load_from_directory(adata[:100].copy(), model_filepath)

    embeddings = None
    n, c = adata.shape[0], 0
    for i in range(0, n, adata_batch_size):
        c += 1
        logging.info(f'starting batch {c} of {int(n/adata_batch_size) + 1}')
        tiny = adata[i:i + adata_batch_size]
        (cell_types, genes, summary, standard_scaler,
            range_scaler, encoder, clf) = parse_module_directory(model_filepath)
        pds = PollockDataset(tiny, batch_size=64, dataset_type='prediction',
                standard_scaler=standard_scaler, range_scaler=range_scaler,
                genes=genes, cell_type_encoder=encoder, cell_types=cell_types)
        cell_embeddings = pm.get_cell_embeddings(pds.prediction_ds)
        df = pd.DataFrame(data=cell_embeddings, index=tiny.obs.index,
                columns=[f'CELL_EMBEDDING_{x+1}' for x in range(cell_embeddings.shape[1])])
        if embeddings is None:
            embeddings = df
        else:
            embeddings = pd.concat((embeddings, df))
    return embeddings


def parse_module_directory(model_filepath):
    """Parses relavent info from the saved module directory"""
    cell_types = np.load(os.path.join(model_filepath, CELL_TYPES_PATH), allow_pickle=True)
    genes = np.load(os.path.join(model_filepath, GENES_PATH), allow_pickle=True)
    summary = json.load(open(os.path.join(model_filepath, MODEL_SUMMARY_PATH)))
    standard_scaler = joblib.load(os.path.join(model_filepath, STANDARD_SCALER_PATH))
    range_scaler = joblib.load(os.path.join(model_filepath, RANGE_SCALER_PATH))
    encoder = joblib.load(os.path.join(model_filepath, ENCODER_PATH))
    clf = joblib.load(os.path.join(model_filepath, CLASSIFIER_PATH))
    return cell_types, genes, summary, standard_scaler, range_scaler, encoder, clf


def load_from_directory(adata, model_filepath, batch_size=64, min_genes_per_cell=None):
    """Convenience method for loading PollockDataset and PollockModel.

    Arguments
    ---------
    adata : anndata.AnnData
        anndata object containing expression of cells to be predicted. adata.X
        attribute must be raw expression counts.
    model_filepath : str
        directory path of a saved pollock model to use for prediction.
    batch_size : int
        minibatch size for BVAE
    min_genes_per_cell : int, None
        filter out cells with less than min_genes_per_cell genes that are
        expressed. Only relavent if planning on prediction straight from 
        10X cellranger outputs or cells have not been filtered already.

    Returns
    -------
    (pollock.models.model.PollockDataset, pollock.models.model.PollockModel)
    """
    (cell_types, genes, summary, standard_scaler,
        range_scaler, encoder, clf) = parse_module_directory(model_filepath)

    prediction_dataset = PollockDataset(adata, batch_size=batch_size, dataset_type='prediction',
            standard_scaler=standard_scaler, range_scaler=range_scaler, genes=genes,
            cell_type_encoder=encoder, cell_types=cell_types)

    latent_dim = summary['model_parameters']['latent_dim']
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
        return tf.data.Dataset.from_tensor_slices(adata.X.toarray()).batch(n)
    else:
        return tf.data.Dataset.from_tensor_slices(adata.X).batch(n)

class PollockModel(object):
    def __init__(self, class_names, input_shape, model=None, learning_rate=1e-4,
            summary=None, alpha=.1, latent_dim=100, clf=None):
        """Model for cell type prediction.

        For usage examples see https://github.com/ding-lab/pollock/tree/master/examples

        Arguments
        ---------
        class_names : Collection
            List of cell types that can be predicted. Index of cell type must
            match the integer class. The easiest way to get this list is
            to get cell types stored in PollockDataset.cell_types
        input_shape : int
            Shape of input layer for BVAE. Will be the same of the number of
            genes in training dataset.
        model : tensorflow.keras.Model, None
            If model is provided, then a PollockModel will be initialized
            from the given tensorflow model. Only used when loading a saved
            PollockModel
        learning_rate : float
            Learning rate for BVAE
        summary : dict, None
            metadata for the pollock model. Typically only set by
            load_from_directory convienince method.
        alpha : float
            alpha for BVAE loss function
        latent_dim : int
            size of latent dimension of BVAE
        clf: sklearn.ensemble.RandomForest, None
            intialize the classifier as the given Random Forest Model.
            Only used if loading a PollockModel.

        Returns
        -------
        pollock.models.model.PollockModel
        """
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
        self.cell_type_train_f1 = {c:[] for c in self.class_names}
        self.cell_type_val_f1 = {c:[] for c in self.class_names}



    def get_cell_embeddings(self, ds):
        """Return cell embeddings for given tf dataset"""
        embeddings = None
        for X in ds:
            mean, logvar = self.model.encode(X)
            emb = mean
            if embeddings is None:
                embeddings = emb
            else:
                embeddings = np.concatenate((embeddings, emb), axis=0)

        return embeddings

    def get_umap_cell_embeddings(self, ds):
        """Get UMAP dim reduction of cell embeddings from
        the given tf dataset"""
        embeddings = self.get_cell_embeddings(ds)
        return umap.UMAP().fit_transform(embeddings)

    def get_cell_type_loss(self, cell_adata, n_per_cell_type):
        """Returns BVAE loss per cell type"""
        ## do training
        if not cell_adata.shape[0]: return 0.0
        # if we are sufficiently small just use the whole thing
        if cell_adata.shape[0] < 10:
            cell_ids = cell_adata.obs.index
        else:
            cell_ids = np.asarray(
                random.sample(list(cell_adata.obs.index), min(cell_adata.shape[0], n_per_cell_type)))

        if 'sparse' in str(type(cell_adata.X)).lower():
            X = cell_adata[cell_ids].X.toarray()
        else:
            X = cell_adata[cell_ids].X
        loss = tf.keras.metrics.Mean()
        loss(compute_loss(self.model, tf.convert_to_tensor(X), alpha=self.alpha))
        return loss.result().numpy()

    def get_cell_type_accuracy(self, X, y, clf=None):
        """Returns accuracy per cell type of random forest classifier"""
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

    def fit(self, pollock_dataset, epochs=10, max_metric_batches=2, metric_epoch_interval=1,
            metric_n_per_cell_type=50):
        """Fit the PollockModel

        Arguments
        ---------
        pollock_dataset : pollock.models.model.PollockDataset
            PollockDataset to use to fit the PollockModel. The PollockDataset should
            be of type 'training'
        epochs : int
            Number of epochs to train the BVAE for
        max_metric_batches : int
            Number of minibatches to get cell type specific metrics for.
            A larger number will provide metrics for more cells, but will
            make model train more slowly
        metric_epoch_interval : int
            Take cell type specific metrics every metric_epoch_interval epochs
        metric_n_per_cell_type : int
            Pool size for each cell type when taking cell type specific metrics

        Returns
        -------
        Fit PollockModel
        """
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

            if (epoch - 1) % metric_epoch_interval == 0:
                ## get losses for each cell type
                ## randomly taking min(100, n_cells) to keep running time down
                for cell_type in self.class_names:
                    ## do training
                    cell_adata = pollock_dataset.train_adata[pollock_dataset.train_adata.obs[
                            pollock_dataset.cell_type_key]==cell_type]
                    self.cell_type_train_losses[cell_type].append(self.get_cell_type_loss(
                            cell_adata, metric_n_per_cell_type))
    
                    ## do validation
                    cell_adata = pollock_dataset.val_adata[pollock_dataset.val_adata.obs[
                            pollock_dataset.cell_type_key]==cell_type]
                    self.cell_type_val_losses[cell_type].append(self.get_cell_type_loss(
                            cell_adata, metric_n_per_cell_type))
    
                ## do accuracies
                ## training
                adata, _ = balancedish_training_generator(pollock_dataset.train_adata,
                        pollock_dataset.cell_type_key, n_per_cell_type=metric_n_per_cell_type)
                y = pollock_dataset.cell_type_encoder.transform(
                        adata.obs[pollock_dataset.cell_type_key].to_numpy().reshape(
                            (-1, 1))).flatten()
                X = batch_adata(adata, n=pollock_dataset.batch_size)
                clf, report = self.get_cell_type_accuracy(X, y,
                        clf=None)
                self.train_accuracies.append(report.get('accuracy', 0.))
                for cell_type in self.class_names: 
                    self.cell_type_train_f1[cell_type].append(report[cell_type]['f1-score'])
        
                ## validation
                adata, _ = balancedish_training_generator(pollock_dataset.val_adata,
                        pollock_dataset.cell_type_key, n_per_cell_type=metric_n_per_cell_type)
                y = [x if x in self.class_names else self.class_names[0] 
                        for x in adata.obs[pollock_dataset.cell_type_key]]
                y = pollock_dataset.cell_type_encoder.transform(np.asarray(y).reshape(
                        (-1, 1))).flatten()
                X = batch_adata(adata, n=pollock_dataset.batch_size)
                _, report = self.get_cell_type_accuracy(X, y,
                        clf=clf)
                self.val_accuracies.append(report.get('accuracy', 0.))
                for cell_type in self.class_names: 
                    self.cell_type_val_f1[cell_type].append(report[cell_type]['f1-score'])

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

    def predict_pollock_dataset(self, pollock_dataset, labels=False, threshold=0.,
            dataset='prediction'):
        """Return cell type predictions for the given Pollock Dataset.

        Arguments
        ---------
        pollock_dataset : pollock.models.model.PollockDataset
            pollock dataset with cells to be annotated
        labels : bool
            Whether to return cell type annotation or integer class
        threshold : float
            Float between zero and one. If max probability for a cell
            type prediction is below threshold, then cell type will be
            classified as 'unknown'
        dataset : str
            Whether to use dataset in training, validation, or prediction
            attribute.

        Returns
        """
        if dataset == 'prediction':
            prediction_batches = pollock_dataset.prediction_ds
        elif dataset == 'training':
            prediction_batches = pollock_dataset.train_ds
        else:
            prediction_batches = pollock_dataset.val_ds

        if not labels:
            return self.predict(prediction_batches)

        probs = self.predict(prediction_batches)
        output_classes = np.argmax(probs, axis=1).flatten()
        output_probs = np.max(probs, axis=1).flatten()

        output_labels, output_probs = zip(*
                [(pollock_dataset.cell_type_encoder.categories_[0][c], prob)
                if prob > threshold else ('unknown', prob)
                for c, prob in zip(output_classes, output_probs)])

        return output_labels, output_probs, probs

    def predict(self, ds):
        """Return probablities for given tf dataset"""
        X = self.get_cell_embeddings(ds)
        probs = self.clf.predict_proba(X)
        return probs

    def save(self, pollock_training_dataset, filepath, score_train=True,
            score_val=True, metadata=None):
        """Save pollock model to directory

        Arguments
        ---------
        pollock_training_dataset : pollock.models.model.PollockDataset
            training dataset associated with the fitted PollockModel
        filepath : str
            directory location to save pollock model at
        score_train : bool
            whether to score training data and save with PollockModel
        metadata : dict, None
            dictionary storing metadata associated with PollockModel
        generate_metrics : bool
            Whether to generate classification metrics. If validating
            on data with different cell labels training data this can
            be set to true to avoid attempting to compare them, as there
            are some metrics/algorithms that require matching cell type
            labels between the training and prediction data
        """
        ## create directory if does not exist
        if not os.path.isdir(filepath):
            os.mkdir(filepath)

        model_fp = os.path.join(filepath, MODEL_PATH)
        self.model.save_weights(model_fp)
        np.save(os.path.join(filepath, CELL_TYPES_PATH),
                np.asarray(pollock_training_dataset.cell_types))
        np.save(os.path.join(filepath, GENES_PATH),
                np.asarray(pollock_training_dataset.genes))
        joblib.dump(pollock_training_dataset.standard_scaler, os.path.join(filepath,
                STANDARD_SCALER_PATH))
        joblib.dump(pollock_training_dataset.range_scaler, os.path.join(filepath,
                RANGE_SCALER_PATH))
        joblib.dump(pollock_training_dataset.cell_type_encoder, os.path.join(filepath,
                ENCODER_PATH))
        joblib.dump(self.clf, os.path.join(filepath, CLASSIFIER_PATH))

        if metadata is not None:
            d = metadata
        else:
            d = {}

        d['history'] = {
                'train_loss': self.train_losses,
                'validation_loss': self.val_losses,
                'train_accuracy': self.train_accuracies,
                'validation_accuracy': self.val_accuracies,
                'cell_type_train_loss': self.cell_type_train_losses,
                'cell_type_val_loss': self.cell_type_val_losses,
                'cell_type_train_f1': self.cell_type_train_f1,
                'cell_type_val_f1': self.cell_type_val_f1,
                }

        d['model_parameters'] = {
                'alpha': self.alpha,
                'learning_rate': self.lr,
                'latent_dim': self.latent_dim,
                'cell_types': self.class_names
                }

        train_batches = batch_adata(pollock_training_dataset.train_adata, n=100)
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
        """Generate classification report for dataset"""
        probs = self.predict(ds)
        predictions = np.argmax(probs, axis=1).flatten()
        predicted_labels = [self.class_names[i] for i in predictions]
        groundtruth = np.asarray([int(i) for i in y])
        groundtruth_labels = [self.class_names[i] for i in groundtruth]

        report = classification_report(groundtruth,
                predictions, target_names=self.class_names,
                labels=list(range(len(self.class_names))), output_dict=True)

        # if we need to 
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
