library(Seurat)
library(rpollock)

args = commandArgs(trailingOnly=TRUE)

explain_fp = args[1]
background_fp = args[2]
module_fp = args[3]
output_fp = args[4]
predicted_key = args[5]
background_sample_size = args[6]

message('loading explain rds object')
## load data
explain = readRDS((explain_fp))
message('loading background rds object')
## load data
background = readRDS((background_fp))

## get cell labels
if (predicted_key %in% colnames(explain)) {
    message(paste('found ', predicted_key, ' in object metadata'))
    labels = explain[[predicted_key]]
} else if (file.exists(predicted_key)) {
    message(paste(predicted_key, " not found in Seurat object. Attempting to load from file."))
    label_df = read.csv(predicted_key, header=FALSE)
    labels = label_df$V1
} else {
    message(paste('looking for predicted labels in active.ident in object slot'))
    labels = explain@active.ident
}

## save module
df = explain_predictions(explain@assays$RNA@counts, background@assays$RNA@counts, labels,
        module_fp, background_sample_size=as.integer(background_sample_size))

message(paste('writing output to ', output_fp))
write.table(df, file=output_fp, quote=FALSE, sep='\t', col.names=TRUE, row.names=TRUE)
