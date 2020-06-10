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
conda install -c epstorrs pollock==0.0.8
```
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


#### Command line tool
```bash
usage: pollock [-h] [--seurat-rds-filepath SEURAT_RDS_FILEPATH]
               [--scanpy-h5ad-filepath SCANPY_H5AD_FILEPATH]
               [--counts-10x-filepath COUNTS_10X_FILEPATH]
               [--min-genes-per-cell MIN_GENES_PER_CELL]
               [--output-type OUTPUT_TYPE] [--output-prefix OUTPUT_PREFIX]
               source_type module_filepath
```

##### Arguments

source_type
  * Input source type. Possible values are: from_seurat, from_10x, from_scanpy.

  
module_filepath
  * Filepath to module to use for classification. The location of the tumor/tissue module to use for classification. For beta, available modules are stored in katmai at `/diskmnt/Projects/Users/estorrs/pollock/modules`.

###### optional arguments

--seurat-rds-filepath SEURAT_RDS_FILEPATH
  * A saved Seurat RDS object to use for classification. Seurat experiment matrix must be raw expression counts (i.e. not normalized)
  
--scanpy-h5ad-filepath SCANPY_H5AD_FILEPATH
  * A saved .h5ad file to use for classification. scanpy data matrix (.X attribute in the anndata object) must be raw expression counts (i.e. not normalized)
  
--counts-10x-filepath COUNTS_10X_FILEPATH
  * Results of 10X cellranger run to be used for classification. There are two options for inputs: 1) the mtx count directory (typically at outs/raw_feature_bc_matrix), and 2) the .h5 file (typically at outs/raw_feature_bc_matrix.h5).

--min-genes-per-cell MIN_GENES_PER_CELL
  * The minimun number of genes expressed in a cell in order for it to be classified. Only used in 10x mode
  
--output-type OUTPUT_TYPE
  * What output type to write. Valid arguments are scanpy and txt
  
--output-prefix OUTPUT_PREFIX
  * Filepath prefix to write output file.
  
##### example basic usage

###### from 10x output

An example of running the single-cell cesc module with 10x .mtx.gz output folder
```bash
pollock from_10x /diskmnt/Projects/Users/estorrs/pollock/modules/sc_cesc --counts-10x-filepath </filepath/to/cellranger/outs/raw_feature_bc_matrix> --output-prefix output --output-type txt
```

An example of running the single-cell cesc module with 10x .h5 output
```bash
pollock from_10x /diskmnt/Projects/Users/estorrs/pollock/modules/sc_cesc --counts-10x-filepath </filepath/to/cellranger/outs/raw_feature_bc_matrix.h5> --output-prefix output --output-type txt
```

###### from seurat rds object

An example of running the single-cell myeloma module with an rds object
```bash
pollock from_seurat /diskmnt/Projects/Users/estorrs/pollock/modules/sc_myeloma --seurat-rds-filepath </filepath/to/seurat/rds> --output-prefix output --output-type txt
```

##### from scanpy h5ad file

An example of running the single-cell myeloma module with an scanpy .h5ad file
```bash
pollock from_scanpy /diskmnt/Projects/Users/estorrs/pollock/modules/sc_myeloma --scanpy-h5ad-filepath </filepath/to/scanpy/h5ad> --output-prefix output --output-type txt
```

#### example basic usage within a docker container

Docker images are available at dockerhub under the image name estorrs/pollock-cpu. To pull the latest image run the following:
```bash
docker pull estorrs/pollock-cpu:latest
```

When using docker, input and ouput file directories need to be mounted as a volume using the docker -v argument.

An example of running the single-cell cesc module from within a docker container. Sections outlined by <> need to be replaced. Note filepaths in the -v flag must be absolute.

ding lab only: the </path/to/modules/directory/> would be /diskmnt/Projects/Users/estorrs/pollock/modules on katmai
```bash
docker run -v </path/to/directory/with/seurat/rds>:/inputs -v </path/to/output/directory>:/outputs -v </path/to/modules/directory/>:/modules -t estorrs/pollock-cpu pollock from_seurat /modules/sc_myeloma --seurat-rds-filepath /inputs/<filename.rds> --output-prefix /outputs/output --output-type txt
```
  
#### Outputs

There are two possible output types:
  * txt : tab seperated text file
  * scanpy: a .h5ad file that can be loaded with scanpy
  
The following fields will be included in the output: predicted cell type, predicted cell type probability, and probabilities for each potential cell type in the module


## docker
Dockerfiles for running pollock can be found in the `docker/` directory. They can also be pulled from estorrs/pollock-cpu on dockerhub. To pull the latest pollock docker image run the following:
```bash
docker pull estorrs/pollock-cpu
```
To see usage with a docker container see the Usage - command line tool - docker section
