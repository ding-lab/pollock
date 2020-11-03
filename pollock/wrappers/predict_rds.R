#!/usr/bin/Rscript --vanilla

library(Seurat)
library(rpollock)

args = commandArgs(trailingOnly=TRUE)

rds_fp = args[1]
module_dir = args[2]
output_fp = args[3]
output_type = args[4]

## load data
message('reading in RDS object')
sce = readRDS((rds_fp))

## predict data
message('predicting cell types')
predictions = predict_cell_types(sce@assays$RNA@counts, module_dir)

## save output
message(paste('writing output to ', output_fp))
if (output_type == 'txt') {
    write.table(predictions, file=output_fp, quote=FALSE, sep='\t', col.names=TRUE, row.names=TRUE)
} else {
    for (col in colnames(predictions)) {
        sce = AddMetaData(object=sce, metadata=predictions[col], col.name=col)
    }
    saveRDS(sce, file = output_fp)
}
