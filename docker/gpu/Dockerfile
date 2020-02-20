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
RUN apt-get install -y libsm6 libxext6 libxrender-dev git vim

RUN git clone https://github.com/estorrs/dotfiles.git \
    && cp dotfiles/.vimrc /root/.vimrc
    && curl -fLo ~/.vim/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim \

RUN git config --global user.email epstorrs@gmail.com \
    && git config --global user.name Erik Storrs



COPY . /tf/pollock
WORKDIR /tf/pollock


RUN /usr/bin/python3 -m pip install .

## CMD jupyter notebook --no-browser --allow-root --ip 0.0.0.0
