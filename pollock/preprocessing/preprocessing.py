import logging
import os
import re
import warnings

import numpy as np
import pandas as pd
import scipy

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
warnings.filterwarnings('ignore')

PREPROCESSING_DIR = os.path.dirname(os.path.abspath(__file__))
GENE_INDEX = os.path.join(PREPROCESSING_DIR, 'gene_index.npy')

def get_expression_df(fp, chunksize=1000):
    chunked_df = pd.read_csv(fp, sep='\t', chunksize=chunksize)

    sparse = None
    cols = []
    index = []
    for i, df in enumerate(chunked_df):
        count = i * chunksize
        logging.info(f'{count} genes of input expression matrix loaded')

        df = df.set_index(df.columns[0])

        if sparse is None:
            sparse = scipy.sparse.csr_matrix(df.values)
        else:
            sparse = scipy.sparse.vstack((sparse, scipy.sparse.csr_matrix(df.values)))

        cols = df.columns
        index += list(df.index)

    sparse = sparse.transpose()
    # we reverse index and columns because we did a transpose
    sparse_df = pd.DataFrame.sparse.from_spmatrix(data=sparse, columns=index, index=cols)
    sparse_df = sparse_df[list(sorted(sparse_df.columns))]
    
    return sparse_df

def adjust_expression_matrix(input_m, input_genes, model_genes):
    """Adjusts expression matrix so it can be input into model.

    Rows are samples and Cols are genes

    This means removing genes that are unique to the input expression matrix.
    And padding expression matrix for genes that are in model but not input.
    """
    new_expression_m = scipy.sparse.csr_matrix(np.zeros((input_m.shape[0], len(model_genes))))
    input_gene_to_index = {g:i for i, g in enumerate(input_genes)}
    model_gene_to_index = {g:i for i, g in enumerate(model_genes)}

    combined = set(input_genes).intersection(set(model_genes))

    combined_input_idxs = [input_gene_to_index[gene] for gene in combined]
    combined_model_idxs = [model_gene_to_index[gene] for gene in combined]

    new_expression_m[:, combined_model_idxs] = input_m[:, combined_input_idxs]

    return new_expression_m

def get_expression_matrix(expression_fp, chunksize=1000):
    logging.info('reading in input expression matrix')
    df = get_expression_df(expression_fp, chunksize=chunksize)

    input_matrix = df.sparse.to_coo().tocsr()
    
    input_genes = np.asarray(df.columns)
    input_samples = np.asarray(df.index)
    logging.info(f'{len(input_genes)} genes and {len(input_samples)} cells in input expression matrix')

    model_genes = np.load(GENE_INDEX)

    logging.info('transforming input expression matrix')
    expression_matrix = adjust_expression_matrix(input_matrix, input_genes, model_genes)

    return expression_matrix, input_samples, model_genes
