# Pollock

![Image of Pollock](https://github.com/ding-lab/pollock/blob/master/images/polluck.png)

A tool for single cell classification

In Development

## Installation
#### Requirements
* OS:
  * macOS 10.12.6 (Sierra) or later
  * Ubuntu 16.04 or later
  * Windows 7 or later (maybe)
  
* Python3.6 or later

* Note: Unfortunately the only Linux distribution Tensorflow is guaranteed to work on is Ubuntu 16.04 or later. Ding Lab this means that pollock will not run on Katmai or Denali (CentOS) unless you are running it in a docker container. See the docker usage section for details.

#### To install
```bash
git clone https://github.com/ding-lab/pollock.git
pip install -e pollock
```

###### Interoperability with Seurat
Additional installation steps are required for pollock integration with seurat. You must have conda installed to run the below step.
```bash
pollock-setup from_seurat
```

NOTE: It is highly recommended that you install pollock in a virtual environment to avoid package versioning conflicts as pollock has many external dependencies. If you want to install inside a conda environment, the installation would look something like the following:
```bash
git clone https://github.com/ding-lab/pollock.git
conda create -n pollock python=3.7
conda activate
pip install -e pollock
pollock-setup from_seurat
```

##### docker
Dockerfiles for running pollock can be found in the `docker/` directory. They can also be pulled from estorrs/pollock-cpu on dockerhub. To pull the latest pollock docker image run the following:
```bash
docker pull estorrs/pollock-cpu
```
To see usage with a docker container see the Usage - docker section

## Usage
```bash
usage: pollock [-h] [--seurat-rds-filepath SEURAT_RDS_FILEPATH]
               [--scanpy-h5ad-filepath SCANPY_H5AD_FILEPATH]
               [--counts-10x-filepath COUNTS_10X_FILEPATH]
               [--min-genes-per-cell MIN_GENES_PER_CELL]
               [--output-type OUTPUT_TYPE] [--output-prefix OUTPUT_PREFIX]
               source_type module_filepath
```

#### Arguments


source_type
  * Input source type. Possible values are: from_seurat, from_10x, from_scanpy.

  
module_filepath
  * Filepath to module to use for classification. The location of the tumor/tissue module to use for classification. For beta, available modules are stored in katmai at `/diskmnt/Projects/Users/estorrs/pollock/modules`. Available modules at this time are the following: `sc_brca`, `sc_cesc`, `sc_hnsc`, `sc_pdac`, `sc_myeloma` and `sn_ccrcc`. More general purpose modules will be available soon, but for now the available modules are seperated by technology and tumor/tissue type.

###### optional arguments

--seurat-rds-filepath SEURAT_RDS_FILEPATH
  * A saved Seurat RDS object to use for classification. Seurat experiment matrix must be raw expression counts (i.e. not normalized)
  

--scanpy-h5ad-filepath SCANPY_H5AD_FILEPATH
  * A saved .h5ad file to use for classification. scanpy data matrix must be raw expression counts (i.e. not normalized)
  
--counts-10x-filepath COUNTS_10X_FILEPATH
  * Results of 10X cellranger run to be used for classification. There are two options for inputs: 1) the mtx count directory (typically at outs/raw_feature_bc_matrix), and 2) the .h5 file (typically at outs/raw_feature_bc_matrix.h5).

--min-genes-per-cell MIN_GENES_PER_CELL
  * The minimun number of genes expressed in a cell in order for it to be classified. Only used in 10x mode
  
--output-type OUTPUT_TYPE
  * What output type to write. Valid arguments are seurat, scanpy, and txt
  
--output-prefix OUTPUT_PREFIX
  * Filepath prefix to write output file. Only used in 10X mode
  
#### example basic usage

##### from 10x output

An example of running the single-cell cesc module with 10x .mtx.gz output folder
```bash
pollock from_10x /diskmnt/Projects/Users/estorrs/pollock/modules/sc_cesc --counts-10x-filepath </filepath/to/cellranger/outs/raw_feature_bc_matrix> --output-prefix output --output-type txt
```

An example of running the single-cell cesc module with 10x .h5 output
```bash
pollock from_10x /diskmnt/Projects/Users/estorrs/pollock/modules/sc_cesc --counts-10x-filepath </filepath/to/cellranger/outs/raw_feature_bc_matrix.h5> --output-prefix output --output-type txt
```

##### from seurat rds object

An example of running the single-cell myeloma module with an rds object
```bash
pollock from_seurat /diskmnt/Projects/Users/estorrs/pollock/modules/sc_myeloma --seurat-rds-filepath </filepath/to/seurat/rds> --output-prefix output --output-type seurat
```

##### from scanpy h5ad file

An example of running the single-cell myeloma module with an scanpy .h5ad file
```bash
pollock from_scanpy /diskmnt/Projects/Users/estorrs/pollock/modules/sc_myeloma --scanpy-h5ad-filepath </filepath/to/scanpy/h5ad> --output-prefix output --output-type scanpy
```

#### example basic usage within a docker container

Docker images are available at dockerhub under the image name estorrs/pollock-cpu. To pull the latest image run the following:
```bash
docker pull estorrs/pollock-cpu
```

When using docker, input and ouput file directories need to be mounted as a volume using the docker -v argument.

An example of running the single-cell cesc module from within a docker container. Sections outlined by <> need to be replaced. Note filepaths in the -v flag must be absolute.

ding lab only: the </path/to/modules/directory/> would be /diskmnt/Projects/Users/estorrs/pollock/modules on katmai
```bash
docker run -v </path/to/directory/with/seurat/rds>:/inputs -v </path/to/output/directory>:/outputs -v </path/to/modules/directory/>:/modules -t estorrs/pollock-cpu pollock from_seurat /modules/sc_myeloma --seurat-rds-filepath /inputs/<filename.rds> --output-prefix /outputs/output --output-type seurat
```
  
## Outputs

There are three possible output types:
  * txt : tab seperated text file
  * seurat: a .rds seurat object
  * scanpy: a .h5ad file that can be loaded with scanpy
  
The following fields will be included in the output: predicted cell type, predicted cell type probability, and probabilities for each potential cell type in the model

