import math

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report


def show_history(history, from_dict=False):
    """
    Plots tensorflow history of model training
    """
    if not from_dict:
        history = history.history

    acc = history['accuracy']
    val_acc = history['val_accuracy']

    loss = history['loss']
    val_loss = history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

## def get_classification_report(predicted_labels, true_labels, classes):
##     return classification_report(true_labels, predicted_labels,
##                 target_names=classes, output_dict=True)


def get_confusion_matrix(predicted_labels, true_labels, classes, show=True):
    """
    Show confusion matrix for predicted vs true labels.
    
    Returns confusion matrix dataframe
    """
    con_mat = tf.math.confusion_matrix(labels=true_labels, predictions=predicted_labels).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
##     classes = [k for k, v in sorted(d.items(), key=lambda x: x[1])]
    con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)

    if show:
        figure = plt.figure(figsize=(8, 8))
        sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    return con_mat_df

## def get_layer_output(X, model, index=None, name=None):
##     """
##     """
##     if index is not None:
##         layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=index).output)
##     elif name is not None:
##         layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(name=name).output)
##     else:
##         raise RuntimeError("Must specify index or name of layer")
## 
##     return layer_model.predict(X)
## 
## def get_model_embedding(X, model, index=-2):
##     """"""
##     return get_layer_output(X, model, index=index)
## 
## def umap_final_layer(X, pollock_model, n_pca=50, n_umap=2):
##     """"""
##     model = pollock_model.model
##     embedding = PCA(n_pca).fit_transform(get_model_embedding(X, model))
##     embedding = umap.UMAP(n_components=n_umap).fit_transform(embedding)
## 
##     return embedding
## 
## def explain_predictions(pollock_model, inputs, label_index, method='smooth_grad', show=True):
##     model = pollock_model.model
##     explainer = SmoothGrad()
##     # Compute SmoothGrad on VGG16
##     grid = explainer.explain((inputs, None), model, label_index, 20, 1.)
##     plt.imshow(grid)
##     return grid
## 
