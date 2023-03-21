FROM ubuntu:18.04

RUN apt-get update && apt-get install -y vim wget curl git

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /miniconda
ENV PATH="/miniconda/bin:$PATH"

RUN git clone https://github.com/ding-lab/pollock.git 
RUN conda env create -n pollock pytorch torchvision torchaudio cpuonly captum scanpy umap-learn -c pytorch -c conda-forge
ENV PATH="/miniconda/envs/pollock/bin:$PATH"

RUN pip install /pollock

CMD /bin/bash