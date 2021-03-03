FROM ubuntu:18.04

RUN apt-get update && apt-get install -y vim wget curl

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /miniconda
ENV PATH="/miniconda/bin:$PATH"

RUN conda config --add channels defaults
RUN conda config --add channels bioconda
RUN conda config --add channels conda-forge

RUN echo 'hello'

RUN conda install -c epstorrs pollock==0.0.14
RUN pip install 'h5py<3.0.0'

ENV RETICULATE_PYTHON="/miniconda/bin/python"

CMD /bin/bash
