import argparse
import logging

import pandas as pd
import scanpy as sc

import pollock
from pollock import PollockDataset, PollockModel, load_from_directory

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('counts_10x_filepath', type=str,
        help='Results of 10X cellranger run to be used for classification. \
There are two options for inputs: 1) the mtx count directory \
(typically at outs/raw_feature_bc_matrix), and 2) the .h5 file (typically at \
outs/raw_feature_bc_matrix.h5).')
parser.add_argument('module_filepath', type=str,
        help='Filepath to module to use for classification')
parser.add_argument('--output', type=str, default='output.tsv',
        help='Filepath to write output file.')
parser.add_argument('--min-genes-per-cell', type=int, default=200,
        help='The minimun number of genes expressed in a cell in order for it \
to be classified.')

args = parser.parse_args()

def main():
    logging.info('loading in 10x data')

    if args.counts_10x_filepath.split('.')[-1] == 'h5':
        logging.info('reading in .h5 file')
        adata = sc.read_10x_h5(args.counts_10x_filepath)
    else:
        logging.info('reading in .mtx.gz file')
        adata = sc.read_10x_mtx(args.counts_10x_filepath,
                var_names='gene_symbols')

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
