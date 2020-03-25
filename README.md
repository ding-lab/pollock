# Pollock

![Image of Pollock](https://github.com/ding-lab/pollock/blob/master/images/pollock.png)

A tool for single cell classification

In Development

### Installation
##### Requirements
* OS:
  * macOS 10.12.6 (Sierra) or later
  * Ubuntu 16.04 or later
  * Windows 7 or later (maybe)
  
* Python3.6 or later

* Note: Unfortunately the only Linux distribution Tensorflow is guaranteed to work on is Ubuntu 16.04 or later. Ding Lab this means that pollock will not run on Katmai or Denali (CentOS) unless you are running it in a docker container. See the docker usage section for details.

##### To install
```bash
git clone https://github.com/ding-lab/pollock.git
pip install -e pollock
```
NOTE: It is highly recommended that you install pollock in a virtual environment to avoid package versioning conflicts as pollock has many external dependencies. If you want to install inside a conda environment, the installation would look something like the following:
```bash
git clone https://github.com/ding-lab/pollock.git
conda create -n pollock python=3.7
conda activate
pip install -e pollock
```

##### docker
Dockerfiles for running pollock can be found in the `docker/` directory. They can also be pulled from estorrs/pollock-cpu on dockerhub. To pull the latest pollock docker image run the following:
```bash
docker pull estorrs/pollock-cpu
```
To see usage with a docker container see the Usage - docker section

### Usage
```bash
usage: pollock [-h] [--output OUTPUT]
               [--min-genes-per-cell MIN_GENES_PER_CELL]
               cellranger_counts_directory module_filepath
```

##### Arguments

cellranger_counts_directory
  * The 10x counts directory holding the counts matrix file to be predicted (typically at `outs/raw_feature_bc_matrix` in cellranger output directory)
  
module_filepath
  * The location of the tumor/tissue module to use for classification. For beta, available modules are stored in katmai at `/diskmnt/Projects/Users/estorrs/pollock/modules`. Available modules at this time are the following: `sc_brca`, `sc_cesc`, `sc_hnsc`, `sc_pdac`, and `sn_ccrcc`. More general purpose modules will be available soon, but for now the available modules are seperated by technology and tumor/tissue type.
  
--min-genes-per-cell
  * Minimum number of genes a cell must express to be considered for prediction. Cells with less than MIN_GENES_PER_CELL will be filtered out. Default is 200.
  
--output
  * Filepath to write predictions output file.
  
##### example basic usage

An example of running the single-cell cesc module
```bash
pollock --output output.tsv </filepath/to/outs/raw_feature_bc_matrix> /diskmnt/Projects/Users/estorrs/pollock/modules/sc_cesc
```

##### example basic usage within a docker container

An example of running the single-cell cesc module from within a docker container. Note that all filepaths must be absolute filepaths.
```bash
docker run -v </filepath/to/outs/raw_feature_bc_matrix>:/inputs -v </path/to/output_dir>:/outputs -v /diskmnt/Projects/Users/estorrs/pollock/modules:/modules -t estorrs/pollock-cpu pollock --output /outputs/output.tsv /inputs /modules/sc_cesc
```
  
### Outputs

Output is a .tsv file with cell_id, predicted_cell_type, and probability fields

