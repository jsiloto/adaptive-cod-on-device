#FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
#FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
#FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
#FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

FROM nvidia/cuda:11.4.0-runtime-ubuntu20.04

WORKDIR /app

# Setting DEBIAN_FRONTEND=noninteractive allows installation
# of some packages to complete without user input.
# Install Python3.8
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3.8 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" \
    apt-get install -y \
    git \
    wget \
    python3-pip \
    python3-opencv \
    unzip \
    sudo \
    vim

RUN pip install torch
RUN pip install torchvision


#RUN conda install -y jupyter torchvision tensorboard pip matplotlib scipy scikit-learn
RUN pip install jupyter tensorboard pip matplotlib scipy scikit-learn
RUN pip install tensorboardX gdown pycocotools pipenv ptflops wget pandas
RUN pip install pycocotools numpy opencv-python tqdm tensorboard tensorboardX pyyaml webcolors
RUN pip install wandb
RUN pip install gevent gunicorn flask
RUN pip install gradio seaborn
#RUN pip install 'torchvision>=0.7.0'