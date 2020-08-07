# Pollock

![Image of Pollock](https://github.com/ding-lab/pollock/blob/master/images/polluck.png)

Pollock is a tool for single cell classification. Pollock is available in both Python, R, and as a command line tool

In Development

## Installation
#### Requirements
* OS:
  * macOS 10.12.6 (Sierra) or later
  * Ubuntu 16.04 or later
  * Windows 7 or later (not tested)
  
* Python3.6 or later

* Anaconda/Conda
  * Working installation of conda and [bioconda](https://bioconda.github.io/). If you are new to conda and bioconda, we recommend following the getting started page [here](https://bioconda.github.io/user/install.html)

#### To install

pollock is available through the conda package manager.

In addition to the default conda channels, pollock requires bioconda. In particular to ensure proper installation you must have your conda channels set up in the correct order by running the following:
```bash
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
```

To install

```bash
conda install -c epstorrs pollock==0.0.10
```

NOTE: tensorflow requires a fair amount of space to build correctly. In some clusters the tmp/ directory does not have enough space for tensorflow to build. If you run pollock and get an error about tensorflow note being available you will have to install it manually using a directory with enough space (> 2GB should be sufficient).

```bash
TMPDIR=<path/to/directory> pip install --cache-dir=<path/to/directory> --build <path/to/directory> tensorflow==2.1.0
```

After that pollock should work correctly

## Usage

Pollock uses deep learning to make cell type predictions. At it's core, pollock is build upon a deep learning technique called a Beta Variational Autoencoder (BVAE).

With pollock, there are a selection of cell type classification modules that have been trained on a variety of single cell RNA-seq datasets. Any of these modules can be used to classify your single cell data.

Additionally, if you have annotated single cell data, pollock can also be used to train a new module based on the given cell types.

### Modules

There are a variety of modules available for cell type classification. They can be found on the dinglab cluster at `/diskmnt/Projects/Users/estorrs/pollock/modules`.

To list available modules run
```bash
ls /diskmnt/Projects/Users/estorrs/pollock/modules
```

You can also create new modules with pollock (see below)

#### Python

[module training tutorial on pbmc dataset](https://github.com/ding-lab/pollock/blob/master/examples/pbmc_model_training.ipynb)

[prediction with an existing module](https://github.com/ding-lab/pollock/blob/master/examples/pbmc_module_prediction.ipynb)

[module examination](https://github.com/ding-lab/pollock/blob/master/examples/pollock_module_examination.ipynb)

#### R

There is an R library rpollock that comes installed with pollock that allows you to train a module and make predictions directly from R.

Note: rpollock is dependent on the R library reticulate, which will sometimes prompt for a python install location. If this occurs, run the below code to find out the location of your python installation. It will output `<path/to/python/executable>`

```bash
which python3
```

When running R you will need to have this line at the very start of your script (before your library imports)
```R
reticulate::use_python("<path/to/python/executable>")
```

[example usage of rpollock on pbmc3k](https://github.com/ding-lab/pollock/blob/master/examples/rpollock_pbmc_prediction.Rmd)

[This notebook](https://github.com/ding-lab/pollock/blob/master/examples/pollock_module_examination.ipynb) is a python script walking over the information that is contained in each module. Though it is in python, all this information is saved in a json file so everything done in that notebook can also be done in R.

## Command line tool
```bash
usage: pollock [-h] [--module-filepath MODULE_FILEPATH]
               [--seurat-rds-filepath SEURAT_RDS_FILEPATH]
               [--scanpy-h5ad-filepath SCANPY_H5AD_FILEPATH]
               [--cell-type-key CELL_TYPE_KEY] [--alpha ALPHA]
               [--epochs EPOCHS] [--latent-dim LATENT_DIM]
               [--n-per-cell-type N_PER_CELL_TYPE]
               [--counts-10x-filepath COUNTS_10X_FILEPATH]
               [--min-genes-per-cell MIN_GENES_PER_CELL] [--txt-output]
               [--output-prefix OUTPUT_PREFIX]
               mode source_type
```

##### Arguments

mode
  * What task/mode is pollock to perform. Valid arguments are:
    * train
    * predict

source_type
  * Input source type. Possible values are: from_seurat, from_10x, from_scanpy.

  
module_filepath
  * If in prediction mode, this is the filepath to module to use for classification. For beta, available modules are stored in katmai at `/diskmnt/Projects/Users/estorrs/pollock/modules`.
  * If in training mode, this is the filepath where pollock will save the trained module.

###### mode specific arguments

--seurat-rds-filepath SEURAT_RDS_FILEPATH
  * A saved Seurat RDS object to use as input. Raw RNA-seq (i.e. not normalized) counts **must** be stored in @assays$RNA@counts. Note that this is where raw rna-seq counts will be stored by most Seurat single cell workflows by default.
  
--scanpy-h5ad-filepath SCANPY_H5AD_FILEPATH
  * A saved .h5ad file to use as input. scanpy expression matrix (.X attribute in the anndata object) must be raw expression counts (i.e. not normalized)
  
--counts-10x-filepath COUNTS_10X_FILEPATH
  * Can only be used with predict mode. Results of 10X cellranger run to be used for classification. There are two options for inputs: 1) the mtx count directory (typically at outs/raw_feature_bc_matrix), and 2) the .h5 file (typically at outs/raw_feature_bc_matrix.h5).

###### specific to train mode
--cell-type-key CELL_TYPE_KEY
  * The key to use for training the pollock module. The key can be one of the following: 1) A string representing a column in the metadata of the input seurat object or .obs attribute of the scanpy anndata object, or 2) filepath to a .txt file where each line is a cell type label. The number of lines must be equal to the number of cells in the input object. The cell types must also be in the same order as the cells in the input object. By default if the input is a Seurat object pollock will use cell type labels in @active.ident, or if the input is a scanpy anndata object pollock will use the label in .obs["leiden"].
  
--alpha ALPHA
  * This parameter controls how regularized the BVAE is. .0001 is the default. If you increase alpha the cell embeddings are typically more noisy, but also more generalizable. If you decrease alpha the cell embeddings are typically less noisy, but also less generalizable.

--epochs EPOCHS
  * Number of epochs to train the neural net for. Default is 20.

--latent-dim LATENT_DIM
  * Size of hidden layer in the B-VAE. Default is 25.
  
--n-per-cell-type N_PER_CELL_TYPE
  * The number of cells per cell type that should be included in the training dataset. Typically this number will be somewhere between 500-2000. The default is 500. If you have a particular cell type in your dataset that has a low cell count it is usually a good idea not to increase n_per_cell_type too much. A good rule of thumb is that n_per_cell_type should be no greater than the minimum cell type count * 10.


###### optional arguments specific to predict mode

--min-genes-per-cell MIN_GENES_PER_CELL
  * The minimun number of genes expressed in a cell in order for it to be classified. Only used in 10x mode
  
--txt-output TXT_OUTPUT
  * If included output will be written to a tab-seperated .txt file. Otherwise output will be saved in the metadata of the input seurat object (.rds) or scanpy anndata object (.h5ad)
  
--output-prefix OUTPUT_PREFIX
  * Filepath prefix to write output file. Extension will be dependent on the inclusion of --output-txt argument. By default the extension will be the same as the input object type. Default value is "output"
  
#### example basic usage

##### predict mode

An example of cell type prediction on a Seurat .RDS object
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

An example of training a model on a Seurat .RDS object that has cell type labels in @active.idents slot. Note this is where cell type labels are typically stored in Seurat workflows.
```bash
pollock train from_seurat --module-filepath <path_to_write_output_module> --seurat-rds-filepath <filepath_to_RDS_object> 
```

An example of training a model on a Seurat .RDS object that has cell type labels stored in a metadata column named "my_special_cell_types".
```bash
pollock train from_seurat --module-filepath <path_to_write_output_module> --seurat-rds-filepath <filepath_to_RDS_object> --cell-type-key my_special_cell_types

```

An example of training a model on a Seurat .RDS object where cell type labels are in a file.
```bash
pollock train from_seurat --module-filepath <path_to_write_output_module> --seurat-rds-filepath <filepath_to_RDS_object> --cell-type-key <filepath_to_cell_labels>
```


An example of training a model on a Seurat .RDS object with custom model hyperparamters
```bash
pollock train from_seurat --module-filepath <path_to_write_output_module> --seurat-rds-filepath <filepath_to_RDS_object>  --alpha .0001 --epochs 20 --latent-dim 25 --n-per-cell-type 500
```

An example of training a model on a scanpy .h5ad object that has cell type labels stored in a column in .obs named "my_special_cell_types".
```bash
pollock train from_scanpy --module-filepath <path_to_write_output_module> --scanpy-h5ad-filepath <filepath_to_h5ad_object> --cell-type-key my_special_cell_types
```



## docker
Dockerfiles for pollock can be found in the `docker/` directory. They can also be pulled from estorrs/pollock-cpu on dockerhub. To pull the latest pollock docker image run the following:
```bash
docker pull estorrs/pollock-cpu:0.0.10
```

#### example basic usage of comand line tool within a docker container

When using docker, the input and ouput file directories need to be mounted as a volume using the docker -v argument.

Below is an example of predicting cell types from within a docker container. Sections outlined by <> need to be replaced. Note file and directory paths in the -v flag must be absolute. For more examples of how the pollock command line tool is used see the above usage examples.

ding lab only: the </path/to/modules/directory/> would be /diskmnt/Projects/Users/estorrs/pollock/modules on katmai
```bash
docker run -v </path/to/directory/with/seurat/rds>:/inputs -v </path/to/output/directory>:/outputs -v </path/to/modules/directory/>:/modules -t estorrs/pollock-cpu:0.0.10 pollock predict from_seurat --module-filepath /modules/<module_name> --seurat-rds-filepath /inputs/<name_of_seurat_rds_file> --output-prefix /outputs/output
```
