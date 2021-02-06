# CUDA 10.1 requires driver version >= 418.39
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && \
    apt-get -y install apt-utils libopencv-dev cmake git sudo vim software-properties-common

RUN pip install matplotlib tqdm yacs Pillow tensorboardx wandb

RUN export PYTHONPATH=$PWD:PYTHONPATH
