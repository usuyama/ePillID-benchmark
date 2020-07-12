ARG CUDA="10.1"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04


RUN apt-get update && apt-get install -y \
    wget \
    ca-certificates \
    sudo \
    git \
    nano \
    bzip2 \
    libx11-6 \
    libturbojpeg \
    cmake \
    software-properties-common \
    gcc \
    g++

# Create a working directory
RUN mkdir /epillid
RUN mkdir /epillid/env
WORKDIR /epillid

#home dir for all users
ENV HOME=/epillid
RUN chmod 777 /epillid

# make the environment definition available
COPY docker/conda/epillidpy36_env.yml /epillid/env/epillidpy36_env.yml

# Install Miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O ~/miniconda.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p /miniconda \
 && rm ~/miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.7 from the rank order environment 
RUN /miniconda/bin/conda install -y conda-build \
#  && /miniconda/bin/conda create -y --name py36 python=3.6.7 \
 && /miniconda/bin/conda env create -f env/epillidpy36_env.yml \ 
 && /miniconda/bin/conda clean -ya


ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false
