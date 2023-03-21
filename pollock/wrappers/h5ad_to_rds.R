library(Seurat)
library(SeuratDisk)

args = commandArgs(trailingOnly=TRUE)

rds_fp = args[1]
h5_fp = args[2]
h5ad_fp = args[3]

Convert(h5ad_fp, dest = "h5seurat", overwrite = TRUE, verbose = TRUE)
obj <- LoadH5Seurat(h5_fp, verbose = TRUE, meta.data = FALSE, misc = FALSE)
saveRDS(obj, file = rds_fp)
