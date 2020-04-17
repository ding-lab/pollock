library(scater)
library(loomR)
library(Seurat)

args <- commandArgs()
rds_fp = args[[6]]

seurat_exp <- readRDS(rds_fp)

seurat_exp.loom <- as.loom(seurat_exp, filename = args[[7]], verbose = FALSE)

seurat_exp.loom$close_all()

