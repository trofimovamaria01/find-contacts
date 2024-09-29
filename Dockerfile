#FROM python:3.8.6-buster

FROM nvidia/cuda:10.2-devel-ubuntu18.04

RUN apt-get update && \
        apt-get install wget && \
    apt-get install -y curl python3.8 python3.8-distutils && \
    ln -s /usr/bin/python3.8 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    python -m pip install -U pip==20.3.3

ENV PROJECT_ROOT /app

ENV DATA_ROOT /data
ENV TEST_DATA_ROOT /test_data

RUN mkdir $PROJECT_ROOT $DATA_ROOT

RUN wget -r "https://drive.google.com/uc?id=1HvZ3LyKar2DHcY6boUnNWoJA6aB1toAN&export=download&confirm=yes" \
        -O bert_final_model.tar.gz \
        && tar -zxvf bert_final_model.tar.gz \
        && rm -rf bert_final_model.tar.gz

COPY . $PROJECT_ROOT

WORKDIR $PROJECT_ROOT

RUN pip install -r requirements.txt

CMD python lib/run.py
