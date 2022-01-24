library(Seurat)
library(SeuratDisk)

args = commandArgs(trailingOnly=TRUE)

rds_fp = args[1]
h5_fp = args[2]

obj = readRDS(rds_fp)
obj <- DietSeurat(obj, assay = 'RNA')

for (col in colnames(obj@meta.data)) {
    if (!is.null(as.numeric(as.character(obj[[col]][1])))) {
        obj@meta.data[[col]] = as.character(obj@meta.data[[col]])
    }
}

SaveH5Seurat(obj, filename = h5_fp)
Convert(h5_fp, dest = "h5ad")
