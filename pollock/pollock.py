import argparse
import logging

import pandas as pd

import pollock
from pollock.preprocessing.preprocessing import get_expression_matrix
from pollock.model.model import get_default_cell_classifier

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('expression_matrix', type=str,
        help='Filepath of expression matrix. Expression matrix must have ensembl \
gene id for rows and cell id for columns.')
parser.add_argument('--min-confidence-level', type=float,
        default=0., help='Classify cells below this confidence level as unknown.')
parser.add_argument('--output', type=str, default='output.tsv',
        help='Location to write output file.')

args = parser.parse_args()

def main():
    logging.info('beginning preprocessing')
    expression_matrix, samples, genes = get_expression_matrix(args.expression_matrix)

    logging.info('loading cell classification model')
    model = get_default_cell_classifier()
    
    logging.info('beginning cell prediction')
    labels, probs = model.predict(expression_matrix, min_confidence=args.min_confidence_level)

    df = pd.DataFrame.from_dict({'id': samples, 'predicted_cell_type': labels})
    df.to_csv(args.output, sep='\t', index=False)

if __name__ == '__main__':
    main()
