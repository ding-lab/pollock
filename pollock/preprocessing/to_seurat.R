library(scater)
library(loomR)
library(Seurat)

args <- commandArgs()
out_rds_fp = args[[7]]

sce.loom <- connect(filename = args[[6]], mode = "r")

sce.seurat <- as.Seurat(sce.loom)

saveRDS(sce.seurat, out_rds_fp)

sce.loom$close_all()
