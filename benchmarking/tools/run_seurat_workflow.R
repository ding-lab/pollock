#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

train_counts_fp = args[1]
train_labels_fp = args[2]
val_counts_fp = args[3]
val_labels_fp = args[4]
output_fp = args[5]

library(Seurat)
## library(SeuratDisk)
library(ggplot2)
library(patchwork)

write("starting workflow", stderr())

## raw_counts <- read.table(file=train_counts_fp, sep="\t")
raw_counts <- read.table(
    file = train_counts_fp,
#    as.is = TRUE
)
write("is.na sum for training: ", stderr())
write(sum(is.na(raw_counts)), stderr())
write("loading training object", stderr())
train <- CreateSeuratObject(raw_counts, project = "train")
write("finished loading training object", stderr())
label_df = read.csv(train_labels_fp, header=FALSE)
labels = label_df$V1
train <- AddMetaData(train, metadata = labels, col.name="groundtruth")

raw_counts <- read.table(
    file = val_counts_fp,
    as.is = TRUE
)
write("loading val object", stderr())
val <- CreateSeuratObject(raw_counts, project = "val")
## label_df = read.csv(val_labels_fp, header=FALSE)
## print(val_labels_fp)
## labels = label_df$V1
## val <- AddMetaData(val, metadata = labels, col.name="groundtruth")


## Convert(train_fp, dest = "h5seurat", overwrite = TRUE)
## train <- LoadH5Seurat(train_fp)
## Convert(val_fp, dest = "h5seurat", overwrite = TRUE)
## val <- LoadH5Seurat(val_fp)

## train <- ReadH5AD(file = train_fp)
## val <- ReadH5AD(file = val_fp)
## Idents(train) <- "groundtruth"
## Idents(val) <- "groundtruth"

## obj.list <- c(train, val)
## ##names(obj.list) <= c("train", "val")
## ##for (i in names(obj.list)) {
## for (i in length(obj.list)) {
## ##  obj.list[[i]] <- NormalizeData(obj.list[[i]], verbose = FALSE)
##   obj.list[[i]] <- SCTransform(obj.list[[i]], verbose = FALSE)
##   obj.list[[i]] <- FindVariableFeatures(obj.list[[i]], selection.method = "vst", nfeatures = 2500, verbose = FALSE)
##   print(obj.list[[i]])
## }

write('transforming train', stderr())
train <- SCTransform(train, verbose = TRUE)
## print('a')
## train[["percent.mt"]] <- PercentageFeatureSet(train, pattern = "^MT-")
## 
## print('b')
## train <- NormalizeData(train, verbose = FALSE)
## print('c')
## train <- FindVariableFeatures(train, selection.method = "vst", 
##         nfeatures = 2000, verbose = FALSE)
write('transforming val', stderr())
val <- SCTransform(val, verbose = TRUE)
## val <- NormalizeData(val, verbose = FALSE)
## val <- FindVariableFeatures(val, selection.method = "vst", 
##         nfeatures = 2000, verbose = FALSE)


## train <- RunPCA(train, features = VariableFeatures(object = train))
## train <- FindNeighbors(train, dims = 1:30)
## train <- FindClusters(train, resolution = 0.5)
## 
## val <- RunPCA(val, features = VariableFeatures(object = val))
## val <- FindNeighbors(val, dims = 1:30)
## val <- FindClusters(val, resolution = 0.5)



## val <- FindVariableFeatures(val, selection.method = "vst", nfeatures = 2500, verbose = FALSE)
## obj.features <- SelectIntegrationFeatures(object.list = obj.list, nfeatures = 2500)
## obj.list <- PrepSCTIntegration(object.list = obj.list, anchor.features = obj.features)

# This command returns dataset 5.  We can also specify multiple refs. (i.e. c(5,6))
## names(obj.list) <= c('train', 'val')
## reference_dataset <- which(names(obj.list) == "train")
## 
## obj.anchors <- FindIntegrationAnchors(object.list = obj.list, normalization.method = "SCT", 
##         anchor.features = obj.features, reference = reference_dataset)
## obj.integrated <- IntegrateData(anchorset = obj.anchors, normalization.method = "SCT")
## 
## obj.integrated <- RunPCA(object = obj.integrated, verbose = FALSE)
## obj.integrated <- RunUMAP(object = obj.integrated, dims = 1:30)
## 
## 
## 
write("setting active idents in train", stderr())
Idents(train) <- "groundtruth"

write("finding anchors", stderr())
anchors <- FindTransferAnchors(reference = train, query = val, 
        dims = 1:30)
print("starting predictions")
predictions <- TransferData(anchorset = anchors, refdata = train$groundtruth, 
        dims = 1:30)

write.table(predictions, file=output_fp, quote=FALSE, sep='\t', col.names=TRUE, row.names=TRUE)

