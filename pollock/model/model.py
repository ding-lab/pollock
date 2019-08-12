import logging
import os
import warnings

import joblib
import numpy as np
import tensorflow as tf

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
warnings.filterwarnings('ignore')

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(MODEL_DIR, 'weights.h5')
TRAIN_MEAN = os.path.join(MODEL_DIR, 'scaler_mean.npy')
TRAIN_STD = os.path.join(MODEL_DIR, 'scaler_std.npy')
CELL_ENCODER = os.path.join(MODEL_DIR, 'cell_type_encoder.pkl')

def get_tf_model(fp):
    """
    Get tensorflow model from filepath.

    Model must be .h5 file
    """
    return tf.keras.models.load_model(fp)

def normalize_matrix(m, mean=None, std=None):
    """Normalizes expression matrix for model input"""
    sums = np.sum(m, axis=1)
    m = m / sums.reshape((len(sums), 1))

    if mean is None:
        mean = np.average(m, axis=0)
    if std is None:
        std = np.std(m, axis=0)

    m = (m - mean) / std
    
    m[np.isneginf(m)] = 0.
    m[np.isinf(m)] = 0.
    m[np.isnan(m)] = 0.
    
    return m, mean, std

def get_default_cell_classifier():
    train_mean = np.load(TRAIN_MEAN)
    train_std = np.load(TRAIN_STD)
    cell_encoder = joblib.load(CELL_ENCODER)
    return CellClassifier(WEIGHTS, train_mean, train_std, cell_encoder)

class CellClassifier(object):
    """Classifies cell types"""
    def __init__(self, model_fp, train_mean, train_std, cell_encoder):
        """"""
        self.model = get_tf_model(model_fp)
        self.train_mean = train_mean
        self.train_std = train_std
        self.cell_encoder = cell_encoder

    def predict_probs(self, X):
        """
        Returns probabilities classification probabilities for the given inputs
        """
        X, _, _ = normalize_matrix(X, mean=self.train_mean, std=self.train_std)
        return self.model.predict(X)

    def predict(self, X, min_confidence=0., batch_size=5000):
        """
        Predict cell type for the given inputs
        """
        all_labels = np.empty((0))
        all_probs = np.empty((0, len(self.cell_encoder.categories_[0])))
        for i in range(0, X.shape[0], batch_size):
            probs = self.predict_probs(X[i:i + batch_size])
            labels = np.asarray([self.cell_encoder.categories_[0][i]
                    for i in np.argmax(probs, axis=1)])
    
            non_confident_idxs = np.asarray([i for i, conf in enumerate(np.max(probs, axis=1))
                    if conf < min_confidence])
            # change non_confident labels to unknown
            if len(non_confident_idxs):
                labels[non_confident_idxs] = 'unknown'

            all_labels = np.concatenate((all_labels, labels))
            all_probs = np.concatenate((all_probs, probs))

            logging.info(f'processed {all_labels.shape[0]} of {X.shape[0]} cells')

        return all_labels, all_probs
