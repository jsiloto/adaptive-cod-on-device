#FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime


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

RUN conda install -y jupyter torchvision tensorboard pip matplotlib scipy scikit-learn
RUN conda install -c menpo opencv
RUN pip install tensorboardX gdown pycocotools pipenv ptflops wget pandas
RUN pip install pycocotools numpy opencv-python tqdm tensorboard tensorboardX pyyaml webcolors
RUN pip install wandb
RUN conda install -y gevent gunicorn flask
RUN pip install gradio seaborn
RUN pip install 'torchvision>=0.7.0'