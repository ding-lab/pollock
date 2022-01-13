import argparse
import logging

import numpy as np
import pandas as pd
import scanpy as sc

import pollock.utils as utils
import pollock.explain as explain

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('mode', type=str,
        help='What task is pollock to perform. Valid values are the following: \
[train, predict, explain]')

parser.add_argument('source_type', type=str,
        help='Input source type. Possible values are: from_seurat, from_10x, \
from_scanpy')

parser.add_argument('--module-filepath', type=str,
        help='If in predict or explain mode, this is the filepath to module to \
use for classification or explanation. \
If in train mode, this is the filepath to save the trained module.')


## seurat specific parameters
parser.add_argument('--seurat-rds-filepath', type=str,
        help='A saved Seurat RDS object to use for classification. Seurat \
experiment matrix must be raw expression counts (i.e. not normalized)')

## scanpy specific parameters
parser.add_argument('--scanpy-h5ad-filepath', type=str,
        help='A saved .h5ad file to use for classification. scanpy \
data matrix must be raw expression counts (i.e. not normalized)')


########################
## optional arguments ##
########################

## optional arguments for prediction mode
## 10x specific parameters
parser.add_argument('--counts-10x-filepath', type=str,
        help='Results of 10X cellranger run to be used for classification. \
There are two options for inputs: 1) the mtx count directory \
(typically at outs/raw_feature_bc_matrix), and 2) the .h5 file (typically at \
outs/raw_feature_bc_matrix.h5).')
parser.add_argument('--min-genes-per-cell', type=int, default=10,
        help='The minimun number of genes expressed in a cell in order for it \
to be classified. Only used in 10x mode. Default value is 10.')

# other
parser.add_argument('--no-umap', action='store_true',
        help='By default VAE embeddings are transformed via UMAP into 2D space \
and incorporated into predicted object. However, this step can take additional time. \
To prevent include the --no-umap flag. This will speed up prediction time.')
parser.add_argument('--txt-output', action='store_true',
        help='If included output will be written to a tab-seperated .txt file. \
Otherwise output will be saved in the metadata of the input seurat object (.rds) or \
scanpy anndata object (.h5ad)')
parser.add_argument('--output-prefix', type=str, default='output',
        help='Filepath prefix to write output file. Extension will be dependent on the inclusion \
of --output-txt argument. By default the extension will be the same as the input object type. \
Default value is "output"')


# optional arguments for explain mode
parser.add_argument('--background-sample-size', type=int, default=100,
    help='Number of cells to sample as background samples. \
The default of 100 cells is sufficient in most use cases. A larger sample size results in longer \
run times, but increased accuracy.')

# optional arguments for training mode

parser.add_argument('--cell-type-key', type=str, default='cell_type',
        help='The key to use for training the pollock module. \
The key must be a sring representing a column in \
the metadata of the input seurat object or .obs attribute of the scanpy anndata object.')

parser.add_argument('--batch-size', type=int, default=64,
        help='Batch size used for training.')
parser.add_argument('--lr', type=float, default=1e-4,
        help='Max learning rate.')
parser.add_argument('--use-cuda', action='store_true',
        help='If present, gpu will be used for training or prediction.')
parser.add_argument('--kl-scaler', type=float, default=1e-3,
        help='This parameter controls how regularized the VAE is. 1e-3 is the default. \
If increased the cell embeddings are typically more noisy, but typically more generalizable. \
If decreased the cell embeddings are typically less noisy, but typically less generalizable')
parser.add_argument('--zinb-scaler', type=float, default=.5,
        help='Controls how much weight to give VAE reconstruction loss.')
parser.add_argument('--clf-scaler', type=float, default=1.,
        help='Controls how much weight to give classification loss.')
parser.add_argument('--epochs', type=int, default=20,
        help='Number of epochs to train the neural net for. Default is 20.')
parser.add_argument('--latent-dim', type=int, default=64,
        help='Size of hidden layer in the VAE. Default is 64.')
parser.add_argument('--enc-out-dim', type=int, default=128,
        help='Size of layer before latent. Default is 128.')
parser.add_argument('--middle-dim', type=int, default=512,
        help='Size of intermediate linear layers. Default is 512.')
parser.add_argument('--use-all-cells', action='store_true',
        help='Use all inputs for training. Will override --val-ids and --n-per-cell-type')
parser.add_argument('--val-ids', type=str,
        help='If present, argument will override --n-per-cell-type. Specifies which cell ids should be used as validation, the remaining cell ids will be used for training. The filepath must be a text file with one cell ID per line.')
parser.add_argument('--n-per-cell-type', type=int, default=500,
        help='Determines how to split input data into validation and training datasets. The input data will be split into training and validation datasets based on the following methadology. Typically this number will be somewhere between 500-2000. Default is 500. If you have a particular cell type in your dataset that has a low cell count it is usually a good idea not to increase n_per_cell_type too much. A good rule of thumb is that n_per_cell_type should be no greater than the minimum cell type count * 10.')


args = parser.parse_args()


def load_10x():
    """Load 10x data from folder and return anndata obj"""
    logging.info('loading in 10x data')

    if args.counts_10x_filepath.split('.')[-1] == 'h5':
        logging.info('reading in .h5 file')
        return sc.read_10x_h5(args.counts_10x_filepath)
    else:
        logging.info('reading in .mtx.gz file')
        adata = sc.read_10x_mtx(args.counts_10x_filepath,
                var_names='gene_symbols')
        adata.var_names_make_unique()
        sc.pp.filter_cells(adata, min_genes=args.min_genes_per_cell)
        return adata


def load_seurat():
    """Load seurat RDS and convert to scanpy"""
    logging.info(f'loading seurat rds at {args.seurat_rds_filepath}')
    adata = utils.convert_rds(args.seurat_rds_filepath)
    adata.var_names_make_unique()

    return adata

def load_scanpy():
    """Load scanpy h5ad"""
    logging.info(f'loading scanpy h5ad at {args.scanpy_h5ad_filepath}')
    adata = sc.read_h5ad(args.scanpy_h5ad_filepath)
    adata.var_names_make_unique()

    return adata


def check_arguments():
    """Check arguments for obvious issues"""

    available_types = ['from_10x', 'from_seurat', 'from_scanpy']
    if args.source_type not in available_types:
        raise RuntimeError(f'source type: {args.source_type} is not a valid source type.\
 source type must be one of the following: {available_types}')

    if args.source_type == 'from_scanpy' and args.scanpy_h5ad_filepath is None:
        raise RuntimeError(f'When running in from_scanpy \
--scanpy-h5ad-filepath must be defined')
    if args.source_type == 'from_seurat' and args.seurat_rds_filepath is None:
        raise RuntimeError(f'When running in from_seurat \
--seurat-rds-filepath must be defined')
    if args.source_type == 'from_10x' and args.counts_10x_filepath is None:
        raise RuntimeError(f'When running in from_10x \
--counts-10x-filelpath must be defined')


def get_probability_df(adata, model):
    """Return dataframe with labels and probabliities"""
    df = pd.DataFrame.from_dict({
        'cell_id': adata.obs.index.to_list(),
        'predicted_cell_type': adata.obs['predicted_cell_type'],
        'predicted_cell_type_probability': adata.obs['predicted_cell_type_probability']})
    df = df.set_index('cell_id')

    ## add cell probabilities
    prob_df = pd.DataFrame(data=adata.obsm['prediction_probs'],
                           columns=model.classes, index=adata.obs.index.to_list())
    prob_df.columns = [f'probability {c}' for c in prob_df.columns]

    df = pd.concat((df, prob_df), axis=1)

    # add umap if present
    if 'X_umap' in adata.obsm:
        df['UMAP1'] = adata.obsm['X_umap'][:, 0].flatten()
        df['UMAP2'] = adata.obsm['X_umap'][:, 1].flatten()

    return df


def run_predict_cell_types(adata):
    logging.info('predicting cell types')
    model = utils.load_model(args.module_filepath)
    if args.use_cuda:
        model = model.cuda()

    logging.info(f'generate umap: {not args.no_umap}')
    a = utils.predict_adata(model, adata, make_umap=not args.no_umap)
    logging.info('cell types predicted')

    if args.txt_output:
        output_fp = args.output_prefix + '.txt'
        df = get_probability_df(a, model)
        df.to_csv(output_fp, sep='\t')
    elif args.source_type == 'from_seurat':
        utils.save_rds(a, args.output_prefix + '.rds')
    elif args.source_type == 'from_scanpy':
        a.write_h5ad(args.output_prefix + '.h5ad')
    else:
        output_fp = args.output_prefix + '.txt'
        df = get_probability_df(a, model)
        df.to_csv(output_fp, sep='\t')


def run_create_module(adata, n_val=5000):
    logging.info(f'creating training and validation objects')
    if args.use_all_cells:
        logging.info('using all cells')
        val_ids = np.random.choice(
            adata.var.index.to_list(), size=min(n_val, adata.shape[0]), replace=False)
        train, val = adata, adata[val_ids]
    elif args.val_ids is not None:
        logging.info(f'using validation ids at {args.val_ids}')
        val_ids = pd.read_csv(args.val_ids, header=None)[0].to_list()
        pool = set(val_ids)
        train_ids = [i for i in val_ids if i not in pool]
        train, val = adata[train_ids], adata[val_ids]
    else:
        logging.info('creating train/val splits')
        train_ids, rest = utils.get_splits(
            adata, args.cell_type_key, args.n_per_cell_type, oversample=True)
        val_ids, _ = utils.get_splits(
            adata[rest], args.cell_type_key, args.n_per_cell_type, oversample=True)
        train, val = adata[train_ids], adata[val_ids]
    train.obs_names_make_unique()
    val.obs_names_make_unique()

    logging.info(f'train dataset size: {train.shape}, val dataset size: {val.shape}')

    logging.info('creating module')
    utils.train_and_save_model(train, val, args)
    logging.info('module created')


def run_explain(adata):
    background_idxs = np.random.choice(
        adata.obs.index.to_list(),
        size=min(args.background_sample_size, adata.shape[0]), replace=False)
    explain_adata, background_adata = adata, adata[background_idxs]
    logging.info('starting explaination')
    model = utils.load_model(args.module_filepath)
    df = explain.explain_predictions(model, explain_adata, background_adata,
        label_key=args.cell_type_key, device='cuda' if args.use_cuda else 'cpu')
    logging.info('finished explaining')
    df.index.name = 'cell_id'

    output_fp = args.output_prefix + '.txt'
    logging.info(f'writing feature weights to {output_fp}')
    df.to_csv(output_fp, sep='\t', index=True, header=True)


def main():
    check_arguments()
    if args.source_type == 'from_seurat':
        adata = load_seurat()
    elif args.source_type == 'from_scanpy':
        adata = load_scanpy()
    elif args.source_type == 'from_10x':
        adata = load_10x()
    else:
        raise RuntimeError(f'{args.source_type} is not a valid source type. must be from_seurat, from_scanpy, or from_10x')

    if args.mode == 'predict':
        run_predict_cell_types(adata)
    elif args.mode == 'explain':
        run_explain(adata)
    elif args.mode == 'train':
        run_create_module(adata)
    else:
        raise RuntimeError(f'{args.mode} is not a valid mode. must be either predict, explain, or train')


if __name__ == '__main__':
    main()
