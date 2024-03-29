FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime



RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.8 python3-pip \
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



## As per installation instructions @ https://pytorch.org/
#RUN pip3 install torch==1.10.1+cu113 \
#                  torchvision==0.11.2+cu113 \
#                  torchaudio==0.10.1+cu113 \
#                  -f https://download.pytorch.org/whl/cu113/torch_stable.html


RUN pip install tensorboardX \
    gdown pycocotools pipenv \
    ptflops wget pandas \
    pycocotools numpy opencv-python \
    tqdm tensorboard tensorboardX \
    pyyaml webcolors jsonlines \
    gradio seaborn gevent gunicorn flask \
    wandb pyyaml webcolors tensorboard matplotlib \
    scipy scikit-learn jupyter



RUN pip install torch_tb_profiler compressai
RUN pip install apache-tvm

ARG UID
ARG GID
ARG USER
ARG GROUP
RUN groupadd -g $GID $GROUP
RUN useradd -r -s /bin/false -g $GROUP -G sudo -u $UID $USER
RUN mkdir /home/$USER
RUN chmod -R 777 /home/$USER

CMD /bin/bash


#COPY requirements.txt ./requirements.txt
#RUN pip install -r requirements.txt


