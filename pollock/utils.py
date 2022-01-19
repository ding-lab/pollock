import argparse
import logging
import json
import os
import re
import subprocess
import pathlib
import uuid
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import scanpy as sc
from umap import UMAP

from pollock.model import PollockModel, fit_model
from pollock.dataloaders import get_train_dataloaders, get_prediction_dataloader


MODEL_PATH = 'model.pth'
SUMMARY_PATH = 'summary.json'

CONVERT_RDS_SCRIPT = os.path.join(pathlib.Path(__file__).parent.absolute(),
    'wrappers', 'rds_to_h5ad.R')

SAVE_RDS_SCRIPT = os.path.join(pathlib.Path(__file__).parent.absolute(),
    'wrappers', 'h5ad_to_rds.R')

DEFAULT_TRAIN_ARGS = {
    'lr': 1e-4,
    'epochs': 20,
    'batch_size': 64,
    'latent_dim': 64,
    'enc_out_dim': 128,
    'middle_dim': 512,
    'kl_scaler': 1e-3,
    'clf_scaler': 1.,
    'zinb_scaler': .5,
    'use_cuda': False,
    'cell_type_key': 'cell_type',
    'module_filepath': './new_module'
}


def listfiles(folder, regex=None):
    """Return all files with the given regex in the given folder structure"""
    for root, folders, files in os.walk(folder):
        for filename in folders + files:
            if regex is None:
                yield os.path.join(root, filename)
            elif re.findall(regex, os.path.join(root, filename)):
                yield os.path.join(root, filename)


def populate_default_args(args):
    for k, v in DEFAULT_TRAIN_ARGS.items():
        if k not in args:
            args[k] = v

    return args


def cap_list(ls, n=100, split=.8, oversample=True):
    """Cap list at n.
    If n is larger than list size * .8, oversample until you hit n.
    """
    pivot = int(len(ls) * split)
    np.random.shuffle(ls)

    if not oversample:
        return ls[:min(pivot, n)]

    return np.random.choice(ls, size=n)



def get_splits(adata, cell_type_key, n_per_cell_type,
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
    pool = set(train_ids)
    val_ids = np.asarray([x for x in adata.obs.index.to_list() if x not in pool])
##     train_idxs = np.arange(adata.shape[0])[np.isin(np.asarray(adata.obs.index), train_ids)]
##     val_idxs = np.delete(np.arange(adata.shape[0]), train_idxs)

    return train_ids, val_ids


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


def get_opt_and_scheduler(model, epochs, steps_per_epoch, lr=1e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs)

    return opt, scheduler


def create_model(dl, epochs, lr=1e-4,
                 latent_dim=64, enc_out_dim=128, middle_dim=512,
                 kl_scaler=1e-3, clf_scaler=1., zinb_scaler=1.,
                 use_cuda=False):

    model = PollockModel(dl.dataset.adata.var.index.to_list(),
                         dl.dataset.cell_types,
                         latent_dim=latent_dim, enc_out_dim=enc_out_dim,
                         middle_dim=middle_dim, kl_scaler=kl_scaler,
                         clf_scaler=clf_scaler, zinb_scaler=zinb_scaler)
    if use_cuda:
        model = model.cuda()

    opt, scheduler = get_opt_and_scheduler(model, epochs, len(dl), lr=lr)

    return model, opt, scheduler


def save_model(model, directory, metadata=None):
    """
    Save pollock model to directory
    """
    if metadata is None:
        metadata = {}
    Path(directory).mkdir(exist_ok=True, parents=True)

    model_fp = os.path.join(directory, MODEL_PATH)
    torch.save(model.state_dict(), model_fp)

    metadata['genes'] = list(model.genes)
    metadata['classes'] = list(model.classes)

    summary_fp = os.path.join(directory, SUMMARY_PATH)
    json.dump(metadata, open(summary_fp, 'w'))


def train_and_save_model(train, val, args):
    if isinstance(args, argparse.Namespace):
        args = vars(args)

    args = populate_default_args(args)

    logging.info('beginning training')
    logging.info('creating dataloaders')
    train_dl, val_dl = get_train_dataloaders(
        train, val, batch_size=args['batch_size'], label_col=args['cell_type_key'])

    logging.info('creating model')
    model, opt, scheduler = create_model(
        train_dl, args['epochs'], lr=args['lr'],
        latent_dim=args['latent_dim'], enc_out_dim=args['enc_out_dim'],
        middle_dim=args['middle_dim'], kl_scaler=args['kl_scaler'],
        clf_scaler=args['clf_scaler'],
        zinb_scaler=args['zinb_scaler'],
        use_cuda=args['use_cuda'])
    logging.info(f'training dataset size: {len(train_dl.dataset)}, validation dataset size: {len(val_dl.dataset)}, cell types: {model.classes}')

    logging.info('fitting model')
    history = fit_model(model, opt, scheduler, train_dl, val_dl, epochs=args['epochs'])
    logging.info('model fitting finished')

    keep = ['epochs', 'lr', 'latent_dim', 'enc_out_dim', 'middle_dim',
            'kl_scaler', 'clf_scaler', 'zinb_scaler']
    metadata = {k: v for k, v in args.items() if k in keep}
    metadata['history'] = history

    fp = args['module_filepath']
    logging.info(f'saving model to {fp}')
    save_model(model, fp, metadata=metadata)


def load_model(directory):
    model_fp = os.path.join(directory, MODEL_PATH)
    summary_fp = os.path.join(directory, SUMMARY_PATH)
    summary = json.load(open(summary_fp))

    model = PollockModel(
        summary['genes'], summary['classes'],
        latent_dim=summary['latent_dim'], enc_out_dim=summary['enc_out_dim'],
        middle_dim=summary['middle_dim'], kl_scaler=summary['kl_scaler'],
        clf_scaler=summary['clf_scaler'], zinb_scaler=summary['zinb_scaler'])

    model.load_state_dict(torch.load(model_fp))

    return model


def predict_dl(dl, model):
    emb, y_prob, = None, None
    use_cuda = next(model.parameters()).is_cuda
    model.eval()
    with torch.no_grad():
        for i, b in enumerate(dl):
            x, x_raw, sf, y = b['x'], b['x_raw'], b['size_factor'], b['y']
            if use_cuda:
                x, x_raw, sf, y = x.cuda(), x_raw.cuda(), sf.cuda(), y.cuda()

            r = model(x, use_means=True)

            b_emb = r['z'].detach().cpu().numpy()
            b_y_prob = r['y'].detach().cpu().numpy()

            if emb is None:
                emb = b_emb
                y_prob = b_y_prob
            else:
                emb = np.concatenate((emb, b_emb), axis=0)
                y_prob = np.concatenate((y_prob, b_y_prob), axis=0)
    return emb, y_prob


def predict_adata(model, adata, make_umap=True, umap_fit_n=10000, batch_size=1024):
    dl = get_prediction_dataloader(adata, model.genes, batch_size=1024)
    logging.info(f'starting prediction of {dl.dataset.adata.shape[0]} cells')
    emb, y_prob = predict_dl(dl, model)
    a = dl.dataset.adata
    a.obsm['X_emb'] = emb

    if make_umap:
        u = UMAP()
        idxs = np.random.choice(np.arange(a.shape[0]),
                                size=min(umap_fit_n, a.shape[0]), replace=False)
        u.fit(emb[idxs])
        a.obsm['X_umap'] = u.transform(emb)

    a.obsm['prediction_probs'] = y_prob

    a.obs['y_pred'] = [np.argmax(probs) for probs in y_prob]
    a.obs['predicted_cell_type_probability'] = [np.max(probs) for probs in y_prob]
    a.obs['predicted_cell_type'] = [model.classes[np.argmax(probs)]
                                    for probs in y_prob]

    prob_df = pd.DataFrame(data=a.obsm['prediction_probs'],
                           columns=model.classes, index=a.obs.index.to_list())
    prob_df.columns = [f'probability {c}' for c in prob_df.columns]

    a.obs = pd.concat((a.obs, prob_df), axis=1)

    return a


def convert_rds(rds_fp):
    h5_fp = str(uuid.uuid4()) + '.h5seurat'
    h5ad_fp = h5_fp.replace('.h5seurat', '.h5ad')
    subprocess.check_output(('Rscript', CONVERT_RDS_SCRIPT, rds_fp, h5_fp))

    adata = sc.read_h5ad(h5ad_fp)

    os.remove(h5_fp)
    os.remove(h5ad_fp)

    return adata


def save_rds(adata, rds_fp):
    h5ad_fp = str(uuid.uuid4()) + '.h5ad'
    h5_fp = h5ad_fp.replace('.h5ad', '.h5seurat')

    adata.write_h5ad(h5ad_fp)

    subprocess.check_output(('Rscript', SAVE_RDS_SCRIPT, rds_fp, h5_fp, h5ad_fp))

    os.remove(h5_fp)
    os.remove(h5ad_fp)
