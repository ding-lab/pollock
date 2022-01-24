# Pollock

![Image of Pollock](https://github.com/ding-lab/pollock/blob/master/images/pollock.png)

Pollock is a deep learning method for single cell classification.

## Installation

#### To install

Pollock is available to run in a Docker image (see below) or can be installed with Conda.

If running without Docker, follow the installation instructions below.

First, download the Pollock repo

```bash
git clone https://github.com/ding-lab/pollock.git
```

Then, create a conda environment from the environmental file within the Pollock repository and activate the conda environment.

```bash
cd pollock
conda env create --file env.yaml
conda activate pollock
pip install .
```

## Usage

Pollock uses deep learning to make cell type predictions. At it's core, pollock is a variational autoencoder (VAE) with a linear classification head.

With pollock, there are a selection of cell type classification models that have been trained on a variety of single cell RNA-seq datasets. Any of these modules can be used to classify your single cell data.

Additionally, if you have annotated single cell data, pollock can also be used to train a new module based on the given cell types.

### Pretrained Models

There are a variety of modules available for cell type classification. The models and training datasets can be found on Zenodo at https://zenodo.org/record/5895221 in pretrained_models.tar.gz and benchmarking_datasets.tar.gz

The following is a list of available pretrained modules:
  * scRNA-seq
    * scRNAseq_brca (trained with disease-specific breast cancer dataset)
    * scRNAseq_cesc (trained with disease-specific cervical cancer dataset)
    * scRNAseq_hnscc (trained with disease-specific head and neck cancer dataset)
    * scRNAseq_melanoma (trained with disease-specific melanoma dataset)
    * scRNAseq_mmy (trained with disease-specific multiple myeloma dataset)
    * scRNAseq_pdac (trained with disease-specific pancreatic cancer dataset)
    * scRNAseq_generalized (trained with all scRNAseq datasets)
    * scRNAseq_brca_panimmune (trained with specific immune cell state breast cancer dataset)
  * snRNAseq
    * snRNAseq_brca (trained with disease-specific breast cancer dataset)
    * snRNAseq_ccrcc (trained with disease-specific kidney cancer dataset)
    * snRNAseq_gbm (trained with disease-specific glioblastoma cancer dataset)
    * snRNAseq_generalized (trained with all snRNAseq datasets)
  * snATACseq
    * snATACseq_brca (trained with disease-specific breast cancer dataset, snATAC-seq data is represented as gene activity)
    * snATACseq_ccrcc (trained with disease-specific kidney cancer dataset, snATAC-seq data is represented as gene activity)
    * snATACseq_brca (trained with disease-specific glioblastoma cancer dataset, snATAC-seq data is represented as gene activity)
    * snATACseq_generalized (trained with all snATACseq datasets)

You can also create new modules with pollock (see training section below)

### Tutorials

#### Python API

[model training and prediction tutorial on pbmc dataset](https://github.com/ding-lab/pollock/blob/master/examples/pbmc_model_training_and_prediction.ipynb)

[prediction and feature score generation with pretrained model](https://github.com/ding-lab/pollock/blob/master/examples/pretrained_model_prediction_and_feature_importances.ipynb)


#### Command line tool
```bash
usage: pollock [-h] [--module-filepath MODULE_FILEPATH] [--seurat-rds-filepath SEURAT_RDS_FILEPATH]
               [--scanpy-h5ad-filepath SCANPY_H5AD_FILEPATH] [--counts-10x-filepath COUNTS_10X_FILEPATH]
               [--min-genes-per-cell MIN_GENES_PER_CELL] [--no-umap] [--txt-output] [--output-prefix OUTPUT_PREFIX]
               [--background-sample-size BACKGROUND_SAMPLE_SIZE] [--cell-type-key CELL_TYPE_KEY] [--batch-size BATCH_SIZE] [--lr LR]
               [--use-cuda] [--kl-scaler KL_SCALER] [--zinb-scaler ZINB_SCALER] [--clf-scaler CLF_SCALER] [--epochs EPOCHS]
               [--latent-dim LATENT_DIM] [--enc-out-dim ENC_OUT_DIM] [--middle-dim MIDDLE_DIM] [--use-all-cells] [--val-ids VAL_IDS]
               [--n-per-cell-type N_PER_CELL_TYPE]
               mode source_type
```

##### Arguments

mode
  * What task/mode is pollock to perform. Valid arguments are:
    * train
    * predict
    * explain

source_type
  * Input source type. Possible values are: from_seurat, from_10x, from_scanpy.

  
module_filepath
  * If in prediction mode, this is the filepath to model to use for classification. Pretrained models can be downloaded here https://zenodo.org/record/5895221 in pretrained_models.tar.gz
  * If in training mode, this is the filepath where pollock will save the trained model.
  * If in explain mode, this is the filepath to the model to use to explain the given pollock predictions.

###### mode specific arguments

--seurat-rds-filepath SEURAT_RDS_FILEPATH
  * A saved Seurat RDS object to use as input. Raw RNA-seq (i.e. not normalized) counts **must** be stored in @assays$RNA@data.
  
--scanpy-h5ad-filepath SCANPY_H5AD_FILEPATH
  * A saved .h5ad file to use as input. scanpy expression matrix (.X attribute in the anndata object) must be raw expression counts (i.e. not normalized)
  
--counts-10x-filepath COUNTS_10X_FILEPATH
  * Can only be used with predict mode. Results of 10X cellranger run to be used for classification. There are two options for inputs: 1) the mtx count directory (typically at outs/raw_feature_bc_matrix), and 2) the .h5 file (typically at outs/raw_feature_bc_matrix.h5).

--min-genes-per-cell-type MIN_GENES_PER_CELL_TYPE
  * The minimun number of genes expressed in a cell in order for it to be classified. Only used in 10x mode. Default value is 10.

###### specific to train mode
--cell-type-key CELL_TYPE_KEY
  * The key to use for training the pollock module. The key must be a sring representing a column in the metadata of the input seurat object or .obs attribute of the scanpy anndata object.')
  
--batch-size BATCH_SIZE
  * Batch size used for training.

--lr LR
  * Max learning rate.

--use-cuda USE_CUDA
  * If present, gpu will be used for training or prediction.

--kl-scaler KL_SCALER
  * This parameter controls how regularized the VAE is. 1e-3 is the default. If increased the cell embeddings are typically more noisy, but typically moregeneralizable. If decreased the cell embeddings are typically less noisy, but typically less generalizable'

--zinb-scaler ZINB_SCALER
  * Controls how much weight to give VAE reconstruction loss.

--clf-scaler CLF_SCALER
  * Controls how much weight to give classification loss.

--epochs EPOCHS
  * Number of epochs to train the neural net for. Default is 20.

--latent-dim LATENT_DIM
  * Size of hidden layer in the VAE. Default is 64.

--encoder-out-dim ENCODER_OUT_DIM
  * Size of layer before latent. Default is 128.

--middle-dim MIDDLE_DIM
  * Size of intermediate linear layers. Default is 512.

--use-all-cells USE_ALL_CELLS
  * Use all inputs for training. Will override --val-ids and --n-per-cell-type

--val-ids VAL_IDS
  * If present, argument will override --n-per-cell-type. Specifies which cell ids should be used as validation, the remaining cell ids will be used for training. The filepath must be a text file with one cell ID per line.

--n-per-cell-type N_PER_CELL_TYPE
  * Is used by default. Determines how to split input data into validation and training datasets. The input data will be split into training and validation datasets based on the following methadology. N_PER_CELL_TYPE cells will be partitioned into training dataset for each cell type. If less than N_PER_CELL_TYPE cells exist for a cell type than cells are oversampled to balance the training dataset. Default is 500.

###### specific to predict mode

--no-umap NO_UMAP
  * By default VAE embeddings are transformed via UMAP into 2D space and incorporated into predicted object. However, this step can take additional time. To prevent include the --no-umap flag. This will speed up prediction time.')
  
--txt-output TXT_OUTPUT
  * If included output will be written to a tab-seperated .txt file. Otherwise output will be saved in the metadata of the input seurat object (.rds) or scanpy anndata object (.h5ad) depending on the input data type.
  
--output-prefix OUTPUT_PREFIX
  * Filepath prefix to write output file. Extension will be dependent on the inclusion of --output-txt argument. By default the extension will be the same as the input object type. Default value is "output"


###### specific to explain mode

--explain-filepath EXPLAIN_FILEPATH
  * Filepath to seurat .rds object or scanpy .h5ad anndata object containing cells to be explained. Expression data must be raw counts (i.e. unnormalized). Larger numbers of cells to explain will mean a longer run time. For reference, running ~100 cells with a background sample size of ~100 cells results in a runtime of approximately 15 minutes. Path to predicted cell type labels is specified by the --predicted-key

--background-filepath BACKGROUND_FILEPATH
  * Filepath to seurat .rds object or scanpy .h5ad anndata object containing cells to use for background samples in model explaination. Expression data must be raw counts (i.e. unnormalized). This object will be sampled to --background-sample-size cells. See --background-sample-size for more details.

###### optional arguments specific to explain mode

--background-sample-size BACKGROUND_SAMPLE_SIZE
  * Number of cells to sample as background samples. The default of 100 cells is sufficient in most use cases.


### example basic usage

##### predict mode

An example of cell type prediction on a Seurat .RDS object and save results to output.rds
```bash
pollock predict from_seurat --module-filepath <path_to_module_directory> --seurat-rds-filepath <filepath_to_RDS_object> --output-prefix output
```

An example of cell type prediction on a Seurat .RDS object, but writing to a txt file instead of an RDS object
```bash
pollock predict from_seurat --module-filepath <path_to_module_directory> --seurat-rds-filepath <filepath_to_RDS_object> --output-prefix output --txt-output
```

An example of cell type prediction on a scanpy .h5ad object
```bash
pollock predict from_scanpy --module-filepath <path_to_module_directory> --scanpy-h5ad-filepath <filepath_to_scanpy_h5ad> --output-prefix output
```

An example of cell type prediction on cellranger output
```bash
pollock predict from_10x --module-filepath <path_to_module_directory> --counts-10x-filepath </filepath/to/cellranger/outs/raw_feature_bc_matrix> --output-prefix output
```

##### train mode

An example of training a model on a Seurat .RDS object that has cell type labels stored as 'cell_type' in the object metadata.
```bash
pollock train from_seurat --module-filepath <path_to_write_output_module> --seurat-rds-filepath <filepath_to_RDS_object> --cell-type-key cell_type

```

An example of training a model on a Seurat .RDS object with some custom model hyperparamters
```bash
pollock train from_seurat --module-filepath <path_to_write_output_module> --seurat-rds-filepath <filepath_to_RDS_object>  --epochs 10 --n-per-cell-type 500
```

An example of training a model on a scanpy .h5ad object that has cell type labels stored in a column in .obs named "cell_type".
```bash
pollock train from_scanpy --module-filepath <path_to_write_output_module> --scanpy-h5ad-filepath <filepath_to_h5ad_object> --cell-type-key cell_type
```

##### explain mode

Note: explain mode can have excessive runtimes for very large numbers of cells, so we recommend downsampling the number of cells in the inputs for faster runtimes.

The explain object contains cells to be explained, the background arguments contains cells to be used as background.

An example of explaining a model for a Seurat .RDS object that has cell type labels in a metadata column named 'cell_type'
```bash
pollock explain from_seurat --explain-filepath <path_to_explain_seurat_object> --background-filepath <path_to_background_seurat_object> --module-filepath <path_to_pollock_module> --output-prefix <path_to_write_output>
```

An example of explaining a model on a Scanpy .h5ad object that has cell type labels in column named 'cell_type' in .obs dataframe.
```bash
pollock explain from_scanpy --explain-filepath <path_to_explain_h5ad> --background-filepath <path_to_background_h5ad> --module-filepath <path_to_pollock_module> --predicted-key cell_type --output-prefix <path_to_write_output>
```

#### Docker
Docker images are available for Pollock. To pull the latest Pollock docker image run the following:
```bash
docker pull estorrs/pollock:0.2.1
```

###### example basic usage of comand line tool within a docker container

When using docker, the input and ouput file directories need to be mounted as a volume using the docker -v argument.

Below is an example of predicting cell types from within a docker container. Sections outlined by <> need to be replaced. Note file and directory paths in the -v flag must be absolute. For more examples of how the pollock command line tool is used see the above usage examples.

Note that if predicting from Seurat RDS object raw, un-normalized counts must be stored in the RNA assay, i.e. the expression data in the object should **not** be normalized.

```bash
docker run -v </path/to/directory/with/seurat/rds>:/inputs -v </path/to/output/directory>:/outputs -v </path/to/modules/directory/>:/modules -t estorrs/pollock:0.2.1 pollock predict from_seurat --module-filepath /modules/<module_name> --seurat-rds-filepath /inputs/<name_of_seurat_rds_file> --output-prefix /outputs/output
```

### Testing

To run Pollock tests navigate to the tests/ directory and run
```bash
pytest -vv test_pollock.py
```
