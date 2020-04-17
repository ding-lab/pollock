import argparse
import logging
import os

import pandas as pd
import scanpy as sc

import pollock
from pollock import PollockDataset, PollockModel, load_from_directory, write_loom

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
parser.add_argument('--output', type=str, default='output.tsv',
        help='Filepath to write output file. Only used in 10X mode')
parser.add_argument('--min-genes-per-cell', type=int, default=10,
        help='The minimun number of genes expressed in a cell in order for it \
to be classified. Only used in 10x mode')


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
    temp_fp = 'temp.loom'
    logging.info('converting seurat to scanpy')
    write_loom(args.seurat_rds_filepath, temp_fp)
    adata = sc.read_loom(temp_fp)
    os.remove(temp_fp)

    return adata

def load_scanpy():
    """Load scanpy h5ad"""
    logging.info('loading scanpy h5ad')
    adata = sc.read_h5ad(args.scanpy_h5ad_filepath)

    return adata

def main():

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
    labels, probs = loaded_pm.predict_pollock_dataset(loaded_pds,
            labels=True)
    logging.info('finish cell prediction')

    df = pd.DataFrame.from_dict({
        'cell_id': list(loaded_pds.prediction_adata.obs.index),
        'predicted_cell_type': labels,
        'probability': probs})

    logging.info(f'writing output to {args.output}')
    df.to_csv(args.output, sep='\t', index=False, header=True)

if __name__ == '__main__':
    main()
