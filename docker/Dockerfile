FROM ubuntu:18.04

RUN apt-get update && apt-get install -y vim wget curl git

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /miniconda
ENV PATH="/miniconda/bin:$PATH"

RUN conda create -n pollock -c pytorch -c conda-forge -y pytorch torchvision torchaudio cpuonly captum scanpy umap-learn r-base
ENV PATH="/miniconda/envs/pollock/bin:$PATH"

RUN Rscript -e 'install.packages(c("seurat", "remotes"), repos="https://cloud.r-project.org")'
RUN Rscript -e 'remotes::install_github("mojaveazure/seurat-disk")'

RUN git clone https://github.com/ding-lab/pollock.git 
RUN pip install /pollock

CMD /bin/bash
