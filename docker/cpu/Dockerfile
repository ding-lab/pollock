FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential vim wget curl libcurl4-openssl-dev python3.6 r-base python3-pip python3-setuptools python3-dev gfortran

## RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
## RUN bash ~/miniconda.sh -b -p /miniconda
## ENV PATH="/miniconda/bin:$PATH"
## 
## RUN conda config --add channels defaults
## RUN conda config --add channels bioconda
## RUN conda config --add channels conda-forge

## RUN conda config --set channel_priority strict

## RUN conda install -y r-seurat>=3.0.2 bioconductor-singlecellexperiment
## RUN conda install -y r-base'<4.0.0,>=3.4.1' libcurl
## ## RUN conda install -y libcurl
## RUN conda install -c r r-curl


RUN R -e "install.packages('curl', repos='http://cran.us.r-project.org')"
RUN R -e "install.packages('Seurat', repos='http://cran.us.r-project.org')"
## RUN R -e "install.packages('BiocManager', repos='http://cran.us.r-project.org')"
## RUN R -e "BiocManager::install('scater')"

## COPY . /pollock
## WORKDIR /pollock
## ## 
## RUN pip3 install -e .
## 
## RUN pip3 install anndata2ri==1.0.2

## 
#RUN pollock-setup from_seurat
## RUN conda install tar
## ENV TAR="/bin/tar"
## RUN conda install -c epstorrs pollock==0.0.9

## ENV PYTHONPATH="/miniconda/lib/python3.7:/miniconda/lib/python3.7/site-packages"
##ENV RETICULATE_PYTHON="/miniconda/bin/python"
## ENV RETICULATE_MINICONDA_ENABLED=FALSE
CMD /bin/bash
