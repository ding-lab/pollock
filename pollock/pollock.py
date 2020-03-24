import argparse
import logging

import pandas as pd
import scanpy as sc

import pollock
from pollock import PollockDataset, PollockModel, load_from_directory

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('cellranger_counts_directory', type=str,
        help='Directory filepath holding results of 10x cellranger run')
parser.add_argument('module_filepath', type=str,
        help='Filepath to module to use for classification')
parser.add_argument('--output', type=str, default='output.tsv',
        help='Filepath to write output file.')

args = parser.parse_args()

def main():
    logging.info('loading in 10x data')
    adata = sc.read_10x_mtx(args.cellranger_counts_directory,
            var_names='gene_symbols')

    logging.info('processing in counts and loading classification module')
    loaded_pds, loaded_pm = load_from_directory(adata, args.module_filepath)

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
