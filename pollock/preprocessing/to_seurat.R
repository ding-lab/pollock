library(scater)
library(loomR)
library(Seurat)

args <- commandArgs()
out_rds_fp = args[[7]]

seurat_exp <- ReadH5AD(file = args[[6]])

saveRDS(seurat_exp, out_rds_fp)
