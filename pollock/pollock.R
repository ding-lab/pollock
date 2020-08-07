#!/usr/bin/Rscript --vanilla


library(Seurat)
library(rpollock)

args = commandArgs(trailingOnly=TRUE)

rds_fp = args[1]
module_dir = args[2]
output_fp = args[3]

## load data
sce = readRDS((rds_fp))

## predict data
predictions = predict_cell_types(sce@assays$RNA@counts, module_dir)

## save output
write.table(predictions, file=output_fp, quote=FALSE, sep='\t', col.names=TRUE, row.names=TRUE)
