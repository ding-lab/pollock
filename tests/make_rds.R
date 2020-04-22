library(dplyr)
library(Seurat)

# Load the PBMC dataset
pbmc.data <- Read10X(data.dir = "~/Downloads/filtered_gene_bc_matrices/hg19/")
# Initialize the Seurat object with the raw (non-normalized data).
pbmc <- CreateSeuratObject(counts = pbmc.data, project = "pbmc3k", min.cells = 3, min.features = 200)
print(pbmc)

## # The [[ operator can add columns to object metadata. This is a great place to stash QC stats
## pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")
## 
## pbmc <- subset(pbmc, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)

pbmc <- FindVariableFeatures(object = pbmc)

saveRDS(pbmc, file = "pbmc_new.rds")

pbmc.loom <- as.loom(pbmc, filename = 'pbmc_new.loom', verbose = FALSE)

pbmc.loom$close_all()
