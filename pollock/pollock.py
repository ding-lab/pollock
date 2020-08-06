import argparse
import logging
import os
import pathlib
import subprocess

import pandas as pd
import scanpy as sc

import pollock
from pollock.models.model import PollockDataset, PollockModel, load_from_directory
from pollock.preprocessing.preprocessing import read_rds

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('mode', type=str,
        help='What task/mode is pollock to perform. Valid values are the following: \
[train, predict]')

parser.add_argument('source_type', type=str,
        help='Input source type. Possible values are: from_seurat, from_10x, \
from_scanpy')

parser.add_argument('--module-filepath', type=str,
        help='If in prediction mode, this is the filepath to module to use for classification. \
If in training mode, this is the filepath to save the trained module.')


## seurat specific parameters
parser.add_argument('--seurat-rds-filepath', type=str,
        help='A saved Seurat RDS object to use for classification. Seurat \
experiment matrix must be raw expression counts (i.e. not normalized)')

## scanpy specific parameters
parser.add_argument('--scanpy-h5ad-filepath', type=str,
        help='A saved .h5ad file to use for classification. scanpy \
data matrix must be raw expression counts (i.e. not normalized)')



## optional arguments
## optional arguments for training mode
parser.add_argument('--cell-type-key', type=str,
        help='The key to use for training the pollock module. \
The key can be one of the following: 1) A string representing a column in \
the metadata of the input seurat object or .obs attribute of the scanpy anndata object, \
or 2) filepath to a .txt file where each line is a cell type label. The number of lines \
must be equal to the number of cells in the input object. The cell types must \
also be in the same order as the cells in the input object. By default if the \
input is a Seurat object pollock will use cell type labels in @active.ident, or \
if the input is a scanpy anndata object pollock will use the label in .obs["leiden"].')
parser.add_argument('--alpha', type=float, default=.0001,
        help='This parameter controls how regularized the BVAE is. .0001 is the default. \
If you increase alpha the cell embeddings are typically more noisy, but also more generalizable. \
If you decrease alpha the cell embeddings are typically less noisy, but also less generalizable')
parser.add_argument('--epochs', type=int, default=20,
        help='Number of epochs to train the neural net for. Default is 20.')
parser.add_argument('--latent-dim', type=int, default=25,
        help='Size of hidden layer in the B-VAE. Default is 25.')
parser.add_argument('--n-per-cell-type', type=int, default=500,
        help='The number of cells per cell type that should be included in the training dataset. \
Typically this number will be somewhere between 500-2000. The default is 500. \
If you have a particular cell type in your dataset that has a low cell count it is usually a \
good idea not to increase n_per_cell_type too much. A good rule of thumb is that n_per_cell_type \
should be no greater than the minimum cell type count * 10.')


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

parser.add_argument('--txt-output', action='store_true',
        help='If included output will be written to a tab-seperated .txt file. \
Otherwise output will be saved in the metadata of the input seurat object (.rds) or \
scanpy anndata object (.h5ad)')
parser.add_argument('--output-prefix', type=str, default='output',
        help='Filepath prefix to write output file. Extension will be dependent on the inclusion \
of --output-txt argument. By default the extension will be the same as the input object type. \
Default value is "output"')


args = parser.parse_args()

## point to location of rpollock scripts
PREDICT_RDS_SCRIPT = os.path.join(pathlib.Path(__file__).parent.absolute(),
                'wrappers', 'predict_rds.R')
TRAIN_RDS_SCRIPT = os.path.join(pathlib.Path(__file__).parent.absolute(),
                'wrappers', 'train_rds.R')

def load_10x():
    """Load 10x data from folder and return anndata obj"""
    logging.info('loading in 10x data')

    if args.counts_10x_filepath.split('.')[-1] == 'h5':
        logging.info('reading in .h5 file')
        return sc.read_10x_h5(args.counts_10x_filepath)
    else:
        logging.info('reading in .mtx.gz file')
        return sc.read_10x_mtx(args.counts_10x_filepath,
                var_names='gene_symbols')


def load_seurat():
    """Load seurat RDS and convert to scanpy"""
    logging.info('loading seurat rds')
    adata = read_rds(args.seurat_rds_filepath)

    return adata

def load_scanpy():
    """Load scanpy h5ad"""
    logging.info('loading scanpy h5ad')
    adata = sc.read_h5ad(args.scanpy_h5ad_filepath)

    return adata

## def save_seurat(adata, output_fp):
##     """Save seurat RDS from adata"""
##     temp_fp = 'temp.h5ad'
##     ## save cell and feature id so loom recongnizes them
##     adata.obs['CellID'] = list(adata.obs.index)
##     adata.var['Gene'] = list(adata.var.index)
## ##     adata.write_loom(temp_fp)
##     adata.write_h5ad(temp_fp)
##     save_rds(temp_fp, output_fp)
##     os.remove(temp_fp)
## 
##     return adata

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

def run_predict_rds_script(rds_fp, module_dir, output_fp, output_type):
    logging.info('running predict rds script')
    subprocess.check_output(('Rscript', PREDICT_RDS_SCRIPT,
            rds_fp, module_dir, output_fp, output_type))
    logging.info('finish predict rds script')

def run_train_rds_script(rds_fp, cell_key, module_dir,
        alpha=.0001, latent_dim=25, n_per_cell_type=500, epochs=20):
    logging.info('running train rds script')
    subprocess.check_output(('Rscript', TRAIN_RDS_SCRIPT,
            rds_fp, module_dir, cell_key,
            str(alpha), str(epochs), str(latent_dim), str(n_per_cell_type)))
    logging.info('finish train rds script')

def main():

    check_arguments()

    if args.mode == 'predict':
        ## run from rds script if seruat input
        if args.source_type == 'from_seurat':
            extension = 'txt' if args.txt_output else 'rds'
            run_predict_rds_script(args.seurat_rds_filepath, args.module_filepath, 
                    f'{args.output_prefix}.{extension}', extension)
        else:
            if args.source_type == 'from_10x':
                adata = load_10x()
            elif args.source_type == 'from_scanpy':
                adata = load_scanpy()
            elif args.source_type == 'from_seurat':
                pass
            else:
                raise RuntimeError(f'{args.source_type} is not a valid source_type')
        
            logging.info('processing in counts and loading classification module')
            min_genes_per_cell = None if args.source_type != 'from_10x' else args.min_genes_per_cell
            print(f'loading model from {args.module_filepath}')
            loaded_pds, loaded_pm = load_from_directory(adata, args.module_filepath,
                    min_genes_per_cell=min_genes_per_cell)
        
            logging.info('start cell prediction')
            labels, label_probs, probs = loaded_pm.predict_pollock_dataset(loaded_pds,
                    labels=True)
            logging.info('finish cell prediction')
        
        
            df = get_probability_df(labels, label_probs, probs, loaded_pds)
        
            if args.txt_output:
                output_fp = f'{args.output_prefix}.txt'
                logging.info(f'writing txt output to {output_fp}')
                df.to_csv(output_fp, sep='\t', index=True, header=True)
            else:
                output_fp = f'{args.output_prefix}.h5ad'
                logging.info(f'writing scanpy anndata in .h5ad format to {output_fp}')
                adata.obs = pd.concat((adata.obs, df), axis=1)
                adata.obs.index.name = 'cell_id'
                adata.write_h5ad(output_fp)
    elif args.mode == 'train':
        ## run from rds script if seruat input
        if args.source_type == 'from_seurat':
            run_train_rds_script(args.seurat_rds_filepath, args.cell_type_key, args.module_filepath,
                    alpha=args.alpha, latent_dim=args.latent_dim, epochs=args.epochs,
                    n_per_cell_type=args.n_per_cell_type)
            logging.info(f'saved module at {args.module_filepath}')
        elif args.source_type == 'from_scanpy':
            adata = load_scanpy()
            logging.info('training pollock module')
            if args.cell_type_key in adata.obs:
                key = args.cell_type_key
            elif os.path.exists(args.cell_type_key):
                logging.info(f'{args.cell_type_key} not found in anndata object. Attempting \
    to load from file.')
                key = 'pollock_training_labels'
                labels = pd.read_csv(args.cell_type_key, header=None, sep='\t')
                labels.columns = ['label']
                if labels.shape[0] != adata.shape[0]:
                    raise RuntimeError(f'Length of cell type labels and anndata object to not match. \
    label length: {len(labels)}, shape of anndata object: {adata.shape}')
                adata.obs[key] = labels['label'].to_list()
            elif 'leiden' in adata.obs:
                key = 'leiden'
            else:
                raise RuntimeError(f'Unable to find cell type key {args.cell_type_key} in anndata, or \
    find load labels from filepath {args.cell_type_key}, or find leiden in .obs')
    
            pds = PollockDataset(adata, cell_type_key=key, n_per_cell_type=args.n_per_cell_type,
                    dataset_type='training')
            pm = PollockModel(pds.cell_types, pds.train_adata.shape[1], alpha=args.alpha,
                    latent_dim=args.latent_dim)
            pm.fit(pds, epochs=args.epochs, max_metric_batches=2, metric_epoch_interval=1,
                    metric_n_per_cell_type=50)
            pm.save(pds, args.module_filepath)
    
            logging.info('finish module training')
            logging.info(f'saved module at {args.module_filepath}')
    
        else:
            raise RuntimeError(f'{args.source_type} is not a valid source_type for train mode')
    else:
        raise RuntimeError(f'Invalid pollock mode of {args.mode}. \
pollock mode must either train or predict')

if __name__ == '__main__':
    main()
