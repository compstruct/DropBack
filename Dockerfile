FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt update \
    && apt install -y software-properties-common\
    && add-apt-repository ppa:deadsnakes/ppa\
    && apt update\
    && apt install -y python3.7-dev\
    && apt clean\
    && apt remove -y software-properties-common\
    && apt autoremove -y

COPY requirements.txt .
RUN apt install -y curl\
  && curl https://bootstrap.pypa.io/get-pip.py | python3.7\
  && pip3 install -r requirements.txt --no-cache-dir\
  && pip install  --no-cache-dir --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali\
  && apt remove -y curl\
  && apt autoremove -y

ENV CHAINER_DATASET_ROOT /local

CMD ["/bin/bash"]