import os
import pathlib
import re
import subprocess

import logging

## CONVERT_SEURAT_SCRIPT = os.path.join(pathlib.Path(__file__).parent.absolute(),
##         'convert_seurat.R')
## TO_SEURAT_SCRIPT = os.path.join(pathlib.Path(__file__).parent.absolute(),
##         'to_seurat.R')
## INSTALL_SEURAT_SCRIPT = os.path.join(pathlib.Path(__file__).parent.absolute(),
##         'install_seurat.R')

def listfiles(folder, regex=None):
    """Return all files with the given regex in the given folder structure"""
    for root, folders, files in os.walk(folder):
        for filename in folders + files:
            if regex is None:
                yield os.path.join(root, filename)
            elif re.findall(regex, os.path.join(root, filename)):
                yield os.path.join(root, filename)


## def write_loom(rds_fp, loom_fp):
##     """Convert seurat rds file to loom file"""
##     output = subprocess.check_output(
##             ('Rscript', CONVERT_SEURAT_SCRIPT, rds_fp, loom_fp))
##     logging.info(output)
## def write_loom(rds_fp, loom_fp):
##     """Convert seurat rds file to loom file"""
##     output = subprocess.check_output(
##             ('Rscript', CONVERT_SEURAT_SCRIPT, rds_fp, loom_fp))
##     command = r'as.SingleCellExperiment(readRDS({rds_fp}))'
## ##     adata = r(r'as.SingleCellExperiment(readRDS("/diskmnt/Projects/Users/estorrs/single_cell_data/myeloma/rds/scRNA/25183_processed_celltype.rds"))')
##     adata = r(command)
##     return adata

def read_rds(rds_fp):
    """Convert seurat rds file to loom file"""
    import anndata2ri
    from rpy2.robjects import r
    anndata2ri.activate()

    r('library(Seurat)')
    command = f'as.SingleCellExperiment(readRDS("{rds_fp}"))'
    adata = r(command)

    return adata

## def save_rds(h5ad_fp, rds_fp):
##     import anndata2ri
##     from rpy2.robjects import r
##     anndata2ri.activate()
## 
##     r('library(Seurat)')
##     command = f'saveRDS(ReadH5AD("{h5ad_fp}"), file="{rds_fp}")'
##     print(command)
##     r(command)

## def save_rds(rds_fp):
##     """Convert seurat rds file to loom file"""
##     import anndata2ri
##     from rpy2.robjects import r
##     anndata2ri.activate()
## 
##     adata_sce = 
## 
##     r('library(Seurat)')
## 
## 
## 
##     command = f'as.SingleCellExperiment(readRDS("{rds_fp}"))'
##     adata = r(command)
## 
##     return adata

## def save_rds(loom_fp, rds_fp):
##     """"""
##     import anndata2ri
##     from rpy2.robjects import r
##     anndata2ri.activate()
## 
##     r('library(Seurat)')
##     command = f'as.SingleCellExperiment(readRDS("{rds_fp}"))'



## def save_rds(loom_fp, rds_fp):
##     """"""
##     output = subprocess.check_output(
##             ('Rscript', TO_SEURAT_SCRIPT, loom_fp, rds_fp))
##     logging.info(output)

## def install_seurat():
##     """"""
##     output = subprocess.check_output(
##             ('Rscript', INSTALL_SEURAT_SCRIPT))
##     logging.info(output)
