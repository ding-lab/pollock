library(Seurat)
library(SeuratData)
library(SeuratDisk)

args = commandArgs(trailingOnly=TRUE)

rds_fp = args[1]
h5_fp = args[2]

obj = readRDS(rds_fp)
obj <- DietSeurat(obj, assay = 'RNA')

SaveH5Seurat(obj, filename = h5_fp)
Convert(h5_fp, dest = "h5ad")
