import argparse
import logging
import os

import pandas as pd
import scanpy as sc

import pollock
from pollock.models.model import PollockDataset, PollockModel, load_from_directory
from pollock.preprocessing.preprocessing import read_rds

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('source_type', type=str,
        help='Input source type. Possible values are: from_seurat, from_10x, \
from_scanpy')
parser.add_argument('module_filepath', type=str,
        help='Filepath to module to use for classification')

## seurat specific parameters
parser.add_argument('--seurat-rds-filepath', type=str,
        help='A saved Seurat RDS object to use for classification. Seurat \
experiment matrix must be raw expression counts (i.e. not normalized)')

## scanpy specific parameters
parser.add_argument('--scanpy-h5ad-filepath', type=str,
        help='A saved .h5ad file to use for classification. scanpy \
data matrix must be raw expression counts (i.e. not normalized)')

## 10x specific parameters
parser.add_argument('--counts-10x-filepath', type=str,
        help='Results of 10X cellranger run to be used for classification. \
There are two options for inputs: 1) the mtx count directory \
(typically at outs/raw_feature_bc_matrix), and 2) the .h5 file (typically at \
outs/raw_feature_bc_matrix.h5).')
parser.add_argument('--min-genes-per-cell', type=int, default=10,
        help='The minimun number of genes expressed in a cell in order for it \
to be classified. Only used in 10x mode')

parser.add_argument('--output-type', type=str, default='txt',
        help='What output type to write. Valid arguments are \
seurat and txt')
parser.add_argument('--output-prefix', type=str, default='output',
        help='Filepath prefix to write output file. Only used in 10X mode')


args = parser.parse_args()

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
        raise RuntimeError(f'When running in from_scanpy mode \
--scanpy-h5ad-filepath must be defined')
    if args.source_type == 'from_seurat' and args.seurat_rds_filepath is None:
        raise RuntimeError(f'When running in from_seurat mode \
--seurat-rds-filepath must be defined')
    if args.source_type == 'from_10x' and args.counts_10x_filepath is None:
        raise RuntimeError(f'When running in from_10x mode \
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

def main():

    check_arguments()

    if args.source_type == 'from_10x':
        adata = load_10x()
    elif args.source_type == 'from_seurat':
        adata = load_seurat()
    elif args.source_type == 'from_scanpy':
        adata = load_scanpy()
    else:
        raise RuntimeError(f'{args.source_type} is not a valid source_type')


    logging.info('processing in counts and loading classification module')
    loaded_pds, loaded_pm = load_from_directory(adata, args.module_filepath,
            min_genes_per_cell=args.min_genes_per_cell)

    logging.info('start cell prediction')
    labels, label_probs, probs = loaded_pm.predict_pollock_dataset(loaded_pds,
            labels=True)
    logging.info('finish cell prediction')


    df = get_probability_df(labels, label_probs, probs, loaded_pds)

    if args.output_type == 'txt':
        output_fp = f'{args.output_prefix}.txt'
        logging.info(f'writing txt output to {output_fp}')
        df.to_csv(output_fp, sep='\t', index=True, header=True)
##     elif args.output_type == 'seurat':
##         output_fp = f'{args.output_prefix}.rds'
##         logging.info(f'writing seurat object to {output_fp}')
##         adata.obs = pd.concat((adata.obs, df), axis=1)
##         adata.obs.index.name = 'cell_id'
##         adata.obs['cell_id'] = list(adata.obs.index)
##         adata.obs = adata.obs.set_index('cell_id')
##         save_seurat(adata, output_fp)
    elif args.output_type == 'scanpy':
        output_fp = f'{args.output_prefix}.h5ad'
        logging.info(f'writing scanpy anndata in .h5ad format to {output_fp}')
        adata.obs = pd.concat((adata.obs, df), axis=1)
        adata.obs.index.name = 'cell_id'
        adata.write_h5ad(output_fp)
    else:
        raise RuntimeError(f'{args.output_type} is not avalid output type')

if __name__ == '__main__':
    main()
