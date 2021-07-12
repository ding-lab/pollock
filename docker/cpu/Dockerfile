FROM ubuntu:18.04

RUN apt-get update && apt-get install -y vim wget curl git

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /miniconda
ENV PATH="/miniconda/bin:$PATH"

RUN git clone https://github.com/ding-lab/pollock.git
RUN conda env create --file pollock/env.yaml

RUN /miniconda3/envs/pollock/bin/R -e "Sys.setenv(TAR = system('which tar', intern = TRUE)); devtools::install_github('https://github.com/estorrs/rpollock')"
ENV RETICULATE_PYTHON="/miniconda/envs/pollock/bin/python"

# make sure we use the right python
ENV PATH="/miniconda/envs/pollock/bin:$PATH"

CMD /bin/bash
