library(Seurat)
library(rpollock)

args = commandArgs(trailingOnly=TRUE)

rds_fp = args[1]
output_fp = args[2]
cell_key = args[3]
alpha = args[4]
epochs = args[5]
latent_dim = args[6]
n_per_cell_type = args[7]

message('loading rds object')
## load data
sce = readRDS((rds_fp))

## get cell labels
if (cell_key %in% colnames(sce)) {
    message(paste('found ', cell_key, ' in object metadata'))
    labels = sce[[cell_key]]
} else if (file.exists(cell_key)) {
    message(paste(cell_key, " not found in Seurat object. Attempting to load from file."))
    label_df = read.csv(cell_key, header=FALSE)
    labels = label_df$V1
} else {
    message(paste('looking for active.ident in object slot'))
    labels = sce@active.ident
}

## save module
message('creating module')
create_module(sce@assays$RNA@counts, labels, output_fp, alpha=as.double(alpha), latent_dim=as.integer(latent_dim),
	      epochs=as.integer(epochs), n_per_cell_type=as.integer(n_per_cell_type))


