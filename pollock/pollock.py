import argparse
import logging

import pandas as pd

import pollock
from pollock import PollockDataset, PollockModel, load_from_directory

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('expression_matrix', type=str,
        help='Filepath of expression matrix. Expression matrix must have ensembl \
gene id for rows and cell id for columns.')
parser.add_argument('module_filepath', type=str,
        help='Filepath to module to use for classification')
parser.add_argument('cellranger_counts_directory', type=str,
        help='Directory filepath holding results of 10x cellranger run')
parser.add_argument('--min-confidence-level', type=float,
        default=0., help='Classify cells below this confidence level as unknown.')
parser.add_argument('--output', type=str, default='output.tsv',
        help='Filepath to write output file.')

args = parser.parse_args()

def main():
    logging.info('loading in 10x data')
    adata = sc.read_10x_mtx(args.cellranger_counts_directory, var_names='gene_symbols')

    logging.info('processing in data and loading classification module')
    loaded_pds, loaded_pm = load_from_directory(adata, args.module_filepath)

    logging.info('start cell prediction')
    
    labels, probs = model.predict(loaded_pds, min_confidence=args.min_confidence_level)

    df = pd.DataFrame.from_dict({
        'id': samples,
        'predicted_cell_type': labels,
        'probability': probs})
    df.to_csv(args.output, sep='\t', index=False, header=True)

if __name__ == '__main__':
    main()
