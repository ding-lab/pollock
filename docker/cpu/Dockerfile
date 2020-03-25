FROM python:3.7

RUN apt-get update && \
    apt-get install -y vim wget curl

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p ./miniconda
ENV PATH="/miniconda/bin:$PATH"

COPY . /pollock
WORKDIR /pollock

RUN pip install .

RUN sed -i 's/\(from pollock.pollock import main\)/sys.path[0] = ""\n\1/' /miniconda/bin/pollock

CMD /bin/bash
