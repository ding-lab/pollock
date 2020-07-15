FROM python:3.6

RUN apt-get update && apt-get install -y vim wget curl

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /miniconda
ENV PATH="/miniconda/bin:$PATH"

RUN conda config --add channels defaults \
  && conda config --add channels bioconda \
  && conda config --add channels conda-forge

## COPY . /pollock
## WORKDIR /pollock
## 
## RUN pip install -e .
## 
## RUN pollock-setup from_seurat
## RUN conda install tar
## ENV TAR="/bin/tar"
RUN conda install -c epstorrs pollock==0.0.9

## ENV PYTHONPATH="/miniconda/lib/python3.7:/miniconda/lib/python3.7/site-packages"
ENV RETICULATE_PYTHON="/miniconda/bin/python"

CMD /bin/bash
