FROM ubuntu:18.04

RUN apt-get update && apt-get install -y vim wget curl git

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /miniconda
ENV PATH="/miniconda/bin:$PATH"

RUN git clone https://github.com/ding-lab/pollock.git 
RUN conda env create --file pollock/env.yaml 

ENV PATH="/miniconda/envs/pollock/bin:$PATH"

RUN R -e "Sys.setenv(TAR = system('which tar', intern = TRUE)); devtools::install_github('https://github.com/estorrs/rpollock')"
RUN R -e "install.packages('SeuratObject', repos = 'https://cloud.r-project.org')"
ENV RETICULATE_PYTHON="/miniconda/envs/pollock/bin/python"

CMD /bin/bash
