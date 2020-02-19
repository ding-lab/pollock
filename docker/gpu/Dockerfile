FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

## ## RUN apt-get update
## ## RUN yes | apt-get install vim wget
## ## 
## ## RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
## ## RUN bash ~/miniconda.sh -b -p ./miniconda
## ## ENV PATH="/miniconda/bin:$PATH"
## RUN apt-get update
## RUN apt-get install -y python3.6
## opencv si weired in docker
RUN apt-get install -y libsm6 libxext6 libxrender-dev git

RUN git clone https://github.com/estorrs/dotfiles.git
RUN cp dotfiles/.vimrc /root/.vimrc

COPY . /tf/pollock
WORKDIR /tf/pollock


RUN /usr/bin/python3 -m pip install .

## CMD jupyter notebook --no-browser --allow-root --ip 0.0.0.0
