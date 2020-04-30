import argparse
import os

import scanpy as sc
import numpy as np

import anndata2ri

from IPython import get_ipython
ipython = get_ipython()

parser = argparse.ArgumentParser()

parser.add_argument('rds_filepath', type=str,
        help='filepath of seurat rds object')
parser.add_argument('output_filepath', type=str,
        help='filepath of output h5ad')

args = parser.parse_args()

# Activate the anndata2ri conversion between SingleCellExperiment and AnnData
anndata2ri.activate()

#Loading the rpy2 extension enables cell magic to be used
#This runs R code in jupyter notebook cells
#%load_ext rpy2.ipython
ipython.magic('load_ext rpy2.ipython')

rds_fp = args.rds_filepath

#%%R -i rds_fp
ipython.magic('R -i rds_fp -o adata')
suppressPackageStartupMessages(library(Seurat))

final = readRDS(file = rds_fp)

#convert the Seurat object to a SingleCellExperiment object
ipython.magic('R -i final -o adata')
adata = as(final, 'SingleCellExperiment')
#adata <- as.SingleCellExperiment(final)

adata.write_h5ad(args.output_filepath)
